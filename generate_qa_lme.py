import os
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from utils import extract_json
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Select LLM mode: "local" or "online"
LLM_MODE = os.getenv("LLM_MODE", "online")

if LLM_MODE == "local":
    # Initialize client for Local LLM
    GENERATION_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-32B")
    llm_client = OpenAI(
        api_key=os.getenv("LOCAL_LLM_API_KEY", "EMPTY"), 
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://0.0.0.0:8088/v1")
    )
    print(f"🚀 Using Local LLM for generation: {llm_client.base_url}, model: {GENERATION_MODEL}")
else:
    # Initialize client for OpenAI (Online)
    GENERATION_MODEL = os.getenv("ONLINE_LLM_MODEL", "gpt-4o-mini")
    llm_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    print(f"🌐 Using OpenAI for generation. model: {GENERATION_MODEL}")

PROMPT_TEMPLATE = """You are an expert data analyst specializing in reading comprehension and fact extraction. You are currently processing part <chunk_index> of a long document. 
Generate <qa_count> Question-Answer (QA) pairs to verify the **newly added key facts** in the provided text chunk. 

To avoid redundancy, here is a list of questions generated from previous chunks. 
Do NOT generate questions that are semantically similar to these: 
--- Existing Questions --- 
<history_questions_answers> 
------------------------ 
# Input Text (Current Chunk) 
<chunk_content> 

# Instructions 
1. **Focus on Incremental Info**: Ask ONLY about **new** events, **new** timelines, **new** dialogue points, or **new** data introduced in this specific chunk. 
2. **Ignore Static Info**: Do not ask about established background facts (e.g., "Who is the protagonist?", "What is his father's name?") unless they change or are first introduced in this chunk. 
3. **Answer Constraints**: 
- Answers must be **short** (entities, dates, short phrases, exact spans). 
- Answers must be objective and verifiable against the text. 
- Paraphrases or semantically equivalent expressions of the information are allowed. 

# Output Format (MUST BE A JSON ARRAY) 
<few_shot>"""

FEW_SHOT = """
[
  {
    "question": "What is the new feature discussed?",
    "answer": "Dark mode"
  }
]
"""

def process_file(input_file, output_file, qa_count=5, max_items=None, min_turns=5):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Check if output file exists and load progress
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    
    processed_ids = {item['question_id'] for item in results}
    
    # Filter out already processed items
    items_to_process = [item for item in data if item['question_id'] not in processed_ids]
    
    # Limit items if max_items is set
    if max_items is not None:
        items_to_process = items_to_process[:max_items]
        print(f"⚠️ Limit applied: Processing only {max_items} items")
    
    for item in tqdm(items_to_process, desc="Processing items"):
        sessions = item['haystack_sessions']
        dates = item.get('haystack_dates', [])
        
        # --- Merge Short Sessions Logic ---
        merged_sessions = []
        merged_dates = []
        
        # Create a copy of sessions to modify during iteration
        # Actually, it's better to build a new list or iterate with index
        # We will use the approach: if current is short, append to next, continue
        
        temp_sessions = [s[:] for s in sessions] # Deep copy of list structure (messages are dicts, ok to share)
        temp_dates = dates[:] if dates else ["Unknown Date"] * len(sessions)
        
        i = 0
        while i < len(temp_sessions):
            current_session = temp_sessions[i]
            current_date = temp_dates[i] if i < len(temp_dates) else "Unknown Date"
            
            # Check if current session is too short (based on number of messages)
            # And it is NOT the last session
            if len(current_session) < min_turns and i < len(temp_sessions) - 1:
                # Merge with NEXT session
                # We prepend current session to next session (chronological order: current then next)
                # Wait, if we are at i, and merge into i+1.
                # sessions[i] comes BEFORE sessions[i+1].
                # So sessions[i+1] = sessions[i] + sessions[i+1]
                temp_sessions[i+1] = current_session + temp_sessions[i+1]
                
                # We do NOT add current to merged_sessions yet.
                # We skip to next iteration, which will process the NEW i+1 (which now contains i)
                i += 1
                continue
            
            # If not short, or is last session, add to final list
            merged_sessions.append(current_session)
            merged_dates.append(current_date)
            i += 1
            
        sessions = merged_sessions
        dates = merged_dates
        # ----------------------------------
        
        all_qa_pairs = []
        
        # We process each session as a chunk
        for i, session in enumerate(sessions):
            # Construct chunk content
            chunk_content = ""
            for msg in session:
                chunk_content += f"{msg['role'].upper()}: {msg['content']}\n"
            
            # Get date for this session
            date_str = "Unknown Date"
            if i < len(dates):
                date_str = dates[i]
            
            # Construct history questions (last 20 pairs to avoid too long prompt)
            history_str = ""
            for qa in all_qa_pairs[-20:]: 
                history_str += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
            
            if not history_str:
                history_str = "None"
            
            # Construct prompt
            prompt = PROMPT_TEMPLATE.replace("<chunk_index>", str(i+1))
            prompt = prompt.replace("<qa_count>", str(qa_count))
            prompt = prompt.replace("<chunk_content>", chunk_content)
            prompt = prompt.replace("<history_questions_answers>", history_str)
            prompt = prompt.replace("<few_shot>", FEW_SHOT)
            
            # Call LLM
            try:
                response = llm_client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                content = response.choices[0].message.content
                
                # Parse JSON
                json_str = extract_json(content)
                try:
                    new_qas = json.loads(json_str)
                    if isinstance(new_qas, list):
                        # Validate structure
                        valid_qas = []
                        for qa in new_qas:
                            if "question" in qa and "answer" in qa:
                                valid_qas.append(qa)
                        all_qa_pairs.extend(valid_qas)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for session {i} of item {item['question_id']}")
                    # print(f"Content: {content}")
            except Exception as e:
                print(f"Error processing session {i} of item {item['question_id']}: {e}")
        
        # Add generated QAs to the item
        new_item = item.copy()
        new_item['generated_qa_pairs'] = all_qa_pairs
        results.append(new_item)
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs for LongMemEval dataset.")
    parser.add_argument("--input", type=str, default="/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json", help="Input JSON file path")
    parser.add_argument("--output", type=str, default="/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_generated_qa.json", help="Output JSON file path")
    parser.add_argument("--qa_num", type=int, default=5, help="Number of QA pairs to generate per chunk")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process (default: all)")
    parser.add_argument("--min_turns", type=int, default=5, help="Minimum number of turns (messages) per chunk to avoid merging")
    
    args = parser.parse_args()
    
    print(f"Reading from {args.input}")
    print(f"Writing to {args.output}")
    print(f"Config: qa_num={args.qa_num}, max_items={args.max_items}, min_turns={args.min_turns}")
    
    # Handle TEST_MODE env var for backward compatibility or convenience
    if os.getenv("TEST_MODE") and args.max_items is None:
        print("⚠️ TEST_MODE detected, setting max_items=2")
        args.max_items = 2
        
    process_file(args.input, args.output, qa_count=args.qa_num, max_items=args.max_items, min_turns=args.min_turns)
