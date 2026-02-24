import os
import json
import time
import argparse
import uuid
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone
from utils import extract_json, parse_messages, MEMREADER_PROMPT

# Load environment variables
load_dotenv()

# Select LLM mode
LLM_MODE = os.getenv("LLM_MODE", "online")
MEMREADER_MODEL = os.getenv("MEMREADER_MODEL", "gpt-4o")

if LLM_MODE == "local":
    llm_client = OpenAI(
        api_key=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"), 
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://0.0.0.0:8088/v1")
    )
    print(f"🚀 Using Local LLM: {llm_client.base_url}, model: {MEMREADER_MODEL}")
else:
    llm_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    print(f"🌐 Using OpenAI: {MEMREADER_MODEL}")

def extract_single_turn(text: str, timestamp: int = None, chat_history: str = "") -> list:
    """
    Extract facts from a single turn using MemReader prompt with history.
    """
    if timestamp is None:
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        today_date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
    
    # Use the MEMREADER_PROMPT from utils
    formatted_prompt = MEMREADER_PROMPT.format(today_date=today_date)
    
    # Construct user input with history
    user_input = ""
    if chat_history:
        user_input += f"Previous Chat History:\n{chat_history}\n\n"
    user_input += f"Current Conversation Turn:\n{text}"
    
    if not user_input.strip():
        return []
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=MEMREADER_MODEL,
                messages=[
                    {"role": "system", "content": formatted_prompt}, 
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}, 
                temperature=0
            )
            raw_content = response.choices[0].message.content
            json_str = extract_json(raw_content)
            fact_objects = json.loads(json_str).get("facts", [])
            
            # Standardize facts
            facts = []
            for fact_obj in fact_objects:
                if fact_obj.get("fact"):
                    facts.append({
                        "text": fact_obj.get("fact", ""),
                        "details": fact_obj.get("details", []),
                        # "timestamp": timestamp
                    })
            return facts
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                print(f"Error extracting facts: {e}")
                return []
    return []

def process_session_turn_by_turn(session, timestamp=None, max_history_turns=10):
    """
    Process a session turn-by-turn to extract facts using chat history.
    Returns the accumulated facts and the full session text.
    """
    all_facts = []
    chat_history = []  # List of turns (list of msgs)
    
    i = 0
    while i < len(session):
        msg = session[i]
        turn = []
        turn.append(msg)
        current_role = msg.get("role")
        
        # Merge next message if it's a response (User -> Assistant or vice versa)
        if current_role != "system" and i + 1 < len(session):
            next_msg = session[i+1]
            next_role = next_msg.get("role")
            if next_role != "system" and next_role != current_role:
                turn.append(next_msg)
                i += 2
            else:
                i += 1
        else:
            i += 1
        
        chat_history.append(turn)
        
        turn_text = parse_messages(turn)
        
        # Get history text (last N turns)
        history_turns = chat_history[:-1][-max_history_turns:]
        history_text = parse_messages([m for t in history_turns for m in t])
        
        # Extract facts for this turn
        turn_facts = extract_single_turn(turn_text, timestamp, history_text)
        
        # Add metadata
        for fact in turn_facts:
            fact["turn_idx"] = len(chat_history)
            all_facts.extend([fact])
            
    return all_facts

def generate_sft_data(input_file, output_file, max_items=None):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_items:
        data = data[:max_items]
        print(f"⚠️ Limit applied: Processing only {max_items} items")
        
    sft_data = []
    
    # Check for existing progress
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                sft_data.append(json.loads(line))
        processed_ids = {item['id'] for item in sft_data if 'id' in item}
        print(f"Resuming from {len(sft_data)} items")
    else:
        processed_ids = set()

    # Flatten sessions from data items
    # Each item in longmemeval has 'haystack_sessions' which is a list of sessions
    # We treat each session as a sample?
    # Or each ITEM as a sample?
    # The user said "input is prompt with chunk". A chunk usually refers to a session in this context.
    # So we iterate over sessions.
    
    # But wait, to resume correctly, we need stable IDs.
    # We can use item_id + session_idx.
    
    tasks = []
    for item in data:
        item_id = item.get('question_id', str(uuid.uuid4()))
        sessions = item['haystack_sessions']
        dates = item.get('haystack_dates', [])
        
        for idx, session in enumerate(sessions):
            task_id = f"{item_id}_session_{idx}"
            if task_id in processed_ids:
                continue
            
            timestamp = None
            if idx < len(dates):
                try:
                    dt = datetime.strptime(dates[idx], "%Y-%m-%d")
                    timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
                except:
                    pass
            
            tasks.append({
                "id": task_id,
                "session": session,
                # "timestamp": timestamp
            })
            
    print(f"Found {len(tasks)} sessions to process")
    
    with open(output_file, 'a') as f:
        for task in tqdm(tasks, desc="Generating SFT Data"):
            session = task['session']
            timestamp = task['timestamp']
            
            # Run the high-quality extraction
            extracted_facts = process_session_turn_by_turn(session, timestamp)
            
            # Construct the input (Chunk text)
            chunk_text = parse_messages(session)
            
            # Format prompt with date
            if timestamp is None:
                today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                today_date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
            
            formatted_prompt = MEMREADER_PROMPT.format(today_date=today_date)
            
            # Construct the sample in Alpaca format (compatible with LLaMA Factory)
            # instruction: System Prompt (Task Definition)
            # input: User Input (Conversation Chunk)
            # output: Model Output (Extracted Facts JSON)
            
            output_json_str = json.dumps({"facts": extracted_facts}, ensure_ascii=False)
            
            sample = {
                "instruction": formatted_prompt,
                "input": chunk_text,
                "output": output_json_str
            }
            
            # Write to file immediately
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data for MemReader distillation.")
    parser.add_argument("--input", type=str, default="/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="/mnt/afs/codes/ljl/Memory-Agent/data/lme/memreader_sft_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process")
    
    args = parser.parse_args()
    
    generate_sft_data(args.input, args.output, args.max_items)
