import os
import json
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from utils import extract_json, MEMREADER_PROMPT
from datetime import datetime

# Load environment variables
load_dotenv()

# Select LLM mode: "local" or "online"
LLM_MODE = os.getenv("LLM_MODE", "online")
MEMREADER_MODEL = os.getenv("MEMREADER_MODEL", "gpt-4o-mini") # Default to gpt-4o-mini for extraction

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

def extract_facts(chunk_content, date_str):
    """
    Extract facts from the chunk using MemReader prompt.
    """
    try:
        # Format the prompt with the provided date
        formatted_prompt = MEMREADER_PROMPT.format(today_date=date_str)
        
        # Call LLM
        response = llm_client.chat.completions.create(
            model=MEMREADER_MODEL,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": chunk_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        
        # Parse JSON
        json_str = extract_json(content)
        result = json.loads(json_str)
        
        # Format facts for QA prompt
        facts = result.get("facts", [])
        formatted_facts = ""
        for i, fact in enumerate(facts):
            if isinstance(fact, str):
                formatted_facts += f"{date_str}: {fact}\n"
            elif isinstance(fact, dict):
                # Handle structured fact with details
                fact_text = fact.get("fact", "")
                details = fact.get("details", [])
                
                fact_str = f"{date_str}: {fact_text}"
                
                if details:
                    if isinstance(details, list):
                        details_str = "; ".join([str(d) for d in details])
                    else:
                        details_str = str(details)
                    fact_str += f" ({details_str})"
                
                formatted_facts += f"{fact_str}\n"
        
        return formatted_facts, facts
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return chunk_content, [] # Fallback to original content

PROMPT_TEMPLATE = """You are an expert data analyst specializing in reading comprehension and fact extraction. You are currently processing part <chunk_index> of a long document. 
 Generate 5 Question-Answer (QA) pairs to verify the **newly added key facts** in the provided text chunk. 
 
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

def process_file(input_file, output_file):
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
    
    if os.getenv("TEST_MODE"):
        items_to_process = items_to_process[:1]
        print("⚠️ TEST MODE: Processing only 1 item")

    for item in tqdm(items_to_process, desc="Processing items"):
        sessions = item['haystack_sessions']
        dates = item.get('haystack_dates', [])
        
        all_qa_pairs = []
        all_extracted_facts = []
        
        # We process each session as a chunk
        for i, session in enumerate(sessions):
            # Construct chunk content
            chunk_content = ""
            for msg in session:
                chunk_content += f"{msg['role'].upper()}: {msg['content']}\n"
            
            # Get date for this session
            date_str = "Unknown Date"
            if i < len(dates):
                try:
                    # Parse date format "2023/05/20 (Sat) 02:21" -> "2023-05-20"
                    date_obj = datetime.strptime(dates[i].split('(')[0].strip(), "%Y/%m/%d")
                    date_str = date_obj.strftime("%Y-%m-%d")
                except:
                    pass
            
            # Extract facts first
            extracted_facts_text, extracted_facts_json = extract_facts(chunk_content, date_str)
            all_extracted_facts.append(extracted_facts_json)
            
            if not extracted_facts_json:
                # If no facts extracted, skip QA generation for this chunk
                continue

            # Construct history questions (last 20 pairs to avoid too long prompt)
            history_str = ""
            for qa in all_qa_pairs[-20:]: 
                history_str += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
            
            if not history_str:
                history_str = "None"
            
            # Construct prompt
            prompt = PROMPT_TEMPLATE.replace("<chunk_index>", str(i+1))
            prompt = prompt.replace("<chunk_content>", extracted_facts_text)
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
        
        # Add generated QAs and extracted facts to the item
        new_item = item.copy()
        new_item['generated_qa_pairs'] = all_qa_pairs
        new_item['extracted_facts'] = all_extracted_facts
        results.append(new_item)
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    input_path = "/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json"
    output_path = "/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_generated_qa.json"
    print(f"Reading from {input_path}")
    print(f"Writing to {output_path}")
    process_file(input_path, output_path)
