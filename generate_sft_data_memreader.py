import os
import json
import time
import argparse
import uuid
import random
import concurrent.futures
import threading
from collections import defaultdict
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
                temperature=0,
                timeout=90  # Add timeout to prevent hanging
            )
            raw_content = response.choices[0].message.content
            json_str = extract_json(raw_content)
            
            # Simple JSON repair attempt
            try:
                fact_objects = json.loads(json_str).get("facts", [])
            except json.JSONDecodeError:
                # Try to close open braces/brackets if truncated
                if json_str.strip().endswith(",") or json_str.strip().endswith("["):
                     pass # Hard to fix easily
                else:
                    # Very naive fix for common truncation
                    fixed_json = json_str.strip()
                    if not fixed_json.endswith("}"):
                        fixed_json += "]}"
                    try:
                        fact_objects = json.loads(fixed_json).get("facts", [])
                    except:
                         raise # Re-raise original error to trigger retry

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
                print(f"Error extracting facts: {str(e)[:100]}")
                return []
    return []

def process_session_turn_by_turn(session, timestamp=None, max_history_turns=10):
    """
    Process a session turn-by-turn to extract facts using chat history.
    Returns the accumulated facts and the full session text.
    Uses internal parallelism to speed up processing of multiple turns.
    """
    all_facts = []
    chat_history = []  # List of turns (list of msgs)
    
    # 1. Pre-calculate all inputs
    turns_inputs = []
    
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
        
        turns_inputs.append({
            "turn_text": turn_text, 
            "history_text": history_text,
            "turn_idx": len(chat_history)
        })

    # 2. Process turns in parallel
    # Limit max workers for internal parallelism
    max_workers = min(10, len(turns_inputs)) if len(turns_inputs) > 0 else 1
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_input = {
            executor.submit(extract_single_turn, inp["turn_text"], timestamp, inp["history_text"]): inp
            for inp in turns_inputs
        }
        
        for future in concurrent.futures.as_completed(future_to_input):
            inp = future_to_input[future]
            try:
                turn_facts = future.result()
                # Add metadata
                for fact in turn_facts:
                    fact["turn_idx"] = inp["turn_idx"]
                    all_facts.extend([fact])
            except Exception as e:
                print(f"Error in turn {inp['turn_idx']}: {e}")

    # Sort facts by turn index to maintain logical order (though not strictly required for set-based evaluation)
    all_facts.sort(key=lambda x: x.get("turn_idx", 0))
            
    return all_facts

def process_task(task):
    """
    Worker function to process a single task.
    """
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
    
    # Construct the sample in Alpaca format
    output_json_str = json.dumps({"facts": extracted_facts}, ensure_ascii=False)
    
    sample = {
        "instruction": formatted_prompt,
        "input": chunk_text,
        "output": output_json_str
    }
    return sample

def generate_sft_data(input_file, output_file, max_items=None, num_threads=4):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_items:
        print(f"🔍 Analyzing distribution for {len(data)} items...")
        
        # Set seed for reproducible sampling across runs
        random.seed(42)
        
        # Group by question type
        type_groups = defaultdict(list)
        for item in data:
            q_type = item.get('question_type', 'unknown')
            type_groups[q_type].append(item)
            
        total_items = len(data)
        sampled_data = []
        
        print(f"📊 Stratified sampling (Target Total: {max_items}):")
        
        # Calculate target count per type
        for q_type, items in type_groups.items():
            ratio = len(items) / total_items
            target_count = int(max_items * ratio)
            
            # Ensure at least 1 item if ratio > 0 but target_count becomes 0 due to rounding
            if target_count == 0 and len(items) > 0:
                 target_count = 1
            
            # Sample without replacement
            # Use random.sample to pick random items
            current_sample = random.sample(items, min(target_count, len(items)))
            sampled_data.extend(current_sample)
            print(f"  - {q_type}: {len(current_sample)} items (Original: {len(items)}, Ratio: {ratio:.1%})")
            
        # If we selected more than max_items (due to rounding up), trim randomly
        if len(sampled_data) > max_items:
             random.shuffle(sampled_data)
             sampled_data = sampled_data[:max_items]
             
        # If we selected fewer than max_items (due to rounding down), fill up randomly
        elif len(sampled_data) < max_items:
             remaining_needed = max_items - len(sampled_data)
             remaining_pool = [item for item in data if item not in sampled_data]
             if remaining_pool:
                 additional = random.sample(remaining_pool, min(remaining_needed, len(remaining_pool)))
                 sampled_data.extend(additional)
        
        # Shuffle final list
        random.shuffle(sampled_data)
        data = sampled_data
        print(f"✅ Sampling complete: Processing {len(data)} items (Sampled from {total_items})")
        
    # Check for existing progress
    if os.path.exists(output_file):
        print(f"Checking existing file: {output_file}")
        valid_data = []
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Check if output has facts
                    output_str = item.get('output', '{}')
                    try:
                        output_obj = json.loads(output_str)
                        facts = output_obj.get('facts', [])
                        
                        if facts: # If facts list is not empty
                            valid_data.append(item)
                        # Else: silently discard empty results
                                
                    except json.JSONDecodeError:
                        pass # Invalid output format
                except json.JSONDecodeError:
                    pass # Invalid line
        
        # Rewrite file with only valid data
        print(f"♻️ Rewriting output file. Keeping {len(valid_data)} items (Discarded empty/invalid results).")
        with open(output_file, 'w') as f:
            for item in valid_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        processed_ids = {item['id'] for item in valid_data if 'id' in item}
        print(f"Resuming from {len(valid_data)} items")
    else:
        processed_ids = set()

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
                "timestamp": timestamp
            })
            
    print(f"Found {len(tasks)} sessions to process")
    
    file_lock = threading.Lock()
    
    with open(output_file, 'a') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Generating SFT Data"):
                task = future_to_task[future]
                try:
                    sample = future.result()
                    # Add ID back to sample for tracking
                    sample['id'] = task['id']
                    
                    # Only write if facts are not empty
                    output_obj = json.loads(sample['output'])
                    if output_obj.get('facts'):
                        with file_lock:
                            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            f.flush()
                    # else:
                        # print(f"Skipping empty result for {task['id']}")
                except Exception as e:
                    print(f"Task {task['id']} generated an exception: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT data for MemReader distillation.")
    parser.add_argument("--input", type=str, default="/mnt/innovator/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="/mnt/innovator/ljl/Memory-Agent/data/lme/memreader_sft_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for parallel processing")
    
    args = parser.parse_args()
    
    generate_sft_data(args.input, args.output, args.max_items, args.num_threads)
