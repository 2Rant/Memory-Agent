import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
from datetime import datetime
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure parallel processing
MAX_WORKERS = 40  # Number of concurrent threads
THREAD_SEMAPHORE = threading.Semaphore(MAX_WORKERS)  # Semaphore to limit concurrent API calls
OUTPUT_LOCK = threading.Lock()  # Lock for thread-safe file writing

# Configure progress tracking
PROGRESS_FILE = "/mnt/afs/codes/ljl/Memory-Agent/data/progress.json"  # File to store processing progress
PROGRESS_LOCK = threading.Lock()  # Lock for thread-safe progress updates

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Modified MEMREADER_PROMPT - timestamp is extracted separately in code
MODIFIED_MEMREADER_PROMPT = """You are an Information Extractor, specialized in accurately extracting and storing relevant facts from conversations. Your primary role is to extract all types of information, including personal preferences and contextual details or any information that is worth remembering.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Here are some few shot examples:

Input:
<User>: Hi.
<Assistant>: Hello! How can I help?
<User>: Just checking in.
Output: {{"facts" : []}}

Input:
<User>: I'm currently on a business trip in Tokyo.
<Assistant>: That sounds busy! How is it going?
<User>: It's okay, but I'm struggling to find a vegetarian restaurant for dinner tonight.
Output: {{"facts" : [
    {{"fact": "User is on a business trip in Tokyo", "details": ["Date: {conversation_date}"]}},
    {{"fact": "User is looking for a vegetarian restaurant", "details": ["Location: Tokyo", "Dietary Restriction: Vegetarian", "Date: {conversation_date} night"]}}
]}}

Input:
<User>: I'm trying to fix my 1967 Ford Mustang.
<Assistant>: What seems to be the problem with it?
<User>: The carburetor is leaking, so I need to order a replacement part.
Output: {{"facts" : [
    {{"fact": "User is trying to fix their 1967 Ford Mustang", "details": ["Issue: Carburetor leaking"]}},
    {{"fact": "User needs to order a replacement carburetor", "details": ["Car: 1967 Ford Mustang", "Issue: Carburetor leaking"]}}
]}}

Input:
<User>: Do you have any recommendations for Christopher Nolan' movies?
<Assistant>: Tenet is quite popular visually. Have you seen it?
<User>: I honestly found Tenet too confusing. But I absolutely loved The Prestige.
Output: {{"facts" : [
    {{"fact": "User found the movie Tenet too confusing", "details": ["Director: Christopher Nolan"]}},
    {{"fact": "User loved the movie The Prestige", "details": ["Director: Christopher Nolan", "Sentiment: Absolutely loved"]}}
]}}

Return the facts in a json format as shown above.

Remember the following:
- Reference Date: The reference date for this conversation is {conversation_date}.
- First-Person Extraction Rule: Treat ALL text input by the user as potential sources of facts. Even if the user provides information in structured formats (like lists, logs, datasets, examples, or code comments), if a sentence contains first-person pronouns ("I", "my", "we") or implies a personal preference/intent, EXTRACT it as a fact. Do not ignore text just because it looks like a list or an example.
- Supplementary Details: Extract context as strict "Category: Value" strings (e.g., "Location: Paris", "Price: $50", "Status: Completed"). The `details` list should provide distinct metadata for filtering, NOT simply repeat words found in the fact. Regarding Dates: ONLY include "Date/Time" in details if it refers to a specific scheduled event, deadline, or future plan
- Context Propagation: Ensure every extracted fact is self-contained. If a shared context (e.g., location, platform, activity) is established anywhere in the messages, explicitly include it in the `details` of all relevant facts, even if not repeated in every sentence.
- ALWAYS resolve relative time expressions (e.g., "yesterday", "next Friday") into absolute ISO dates (YYYY-MM-DD) based on Today's date in the details.
- Do not return anything from the custom few shot example prompts provided above.
- If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
- Create the facts based on the user and assistant messages only. Do not pick anything from the system messages.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts", where each item has a "fact" string and a "details" list of strings.

Following is a conversation between the user and the assistant. Extract all relevant facts from the conversation and return them in the required JSON format.
"""

def read_progress():
    """
    Read processing progress from file
    Returns list of processed instance IDs
    """
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                return progress_data.get('processed_instances', [])
        else:
            return []
    except json.JSONDecodeError:
        print(f"Error parsing progress file. Starting from scratch.")
        return []
    except Exception as e:
        print(f"Error reading progress file: {e}. Starting from scratch.")
        return []

def write_progress(processed_instances):
    """
    Write processing progress to file
    """
    try:
        with PROGRESS_LOCK:
            progress_data = {
                'processed_instances': processed_instances,
                'timestamp': datetime.now().isoformat()
            }
            with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing progress file: {e}")

def parse_and_convert_to_iso(timestamp_str):
    """
    Parse timestamp string and convert to ISO format (YYYY-MM-DD)
    """
    import datetime
    
    # List of common timestamp formats to try
    formats = [
        '%Y-%m-%d %H:%M',          # 2024-01-01 00:00
        '%Y-%m-%d',                 # 2024-01-01
        '%Y/%m/%d (Sat) %H:%M',     # 2023/02/11 (Sat) 13:40
        '%Y/%m/%d',                 # 2023/02/11
        '%B %d, %Y',                # January 1, 2024
        '%d %B %Y',                 # 1 January 2024
        '%m/%d/%Y',                 # 01/01/2024
        '%d/%m/%Y',                 # 01/01/2024
    ]
    
    # Try to parse with each format
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(timestamp_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # If no format matches, try to extract just the year-month-day part
    # This handles cases like "2024-01-01 The user is reading a book"
    match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', timestamp_str)
    if match:
        try:
            dt = datetime.datetime.strptime(match.group(1), '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            try:
                dt = datetime.datetime.strptime(match.group(1), '%Y/%m/%d')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    return "Unknown"

def extract_timestamp_from_chunk(chunk):
    """
    Extract timestamp from chunk header and convert to ISO format (YYYY-MM-DD)
    """
    # Patterns to match different timestamp formats
    patterns = [
        r'\[Dialogue at timestamp (.*?)\]',
        r'\[Dialogue between User and Assistant on (.*?)\]',
        r'\[Event happened on (.*?)\]',
        r'Dialogue happened (.*?):',
        r'Event happened (.*?):'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, chunk)
        if match:
            # Extract and convert to ISO format
            raw_timestamp = match.group(1)
            iso_timestamp = parse_and_convert_to_iso(raw_timestamp)
            return iso_timestamp
    
    return "Unknown"

def process_single_chunk(chunk, client, prompt):
    """
    Process a single chunk: extract timestamp and facts
    """
    with THREAD_SEMAPHORE:  # Limit concurrent API calls
        # Extract timestamp from chunk header
        timestamp = extract_timestamp_from_chunk(chunk)
        
        # Extract facts using modified prompt, passing the timestamp
        facts = get_facts_from_chunk(client, chunk, prompt, timestamp)
        
        # Process and clean facts for this chunk
        chunk_facts = []
        for fact in facts:
            # Remove timestamp from details if it's there (to avoid duplication)
            cleaned_details = [detail for detail in fact["details"] if not detail.startswith("Timestamp:")]
            
            chunk_fact = {
                "fact": fact["fact"],
                "details": cleaned_details,
                "timestamp": timestamp
            }
            chunk_facts.append(chunk_fact)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
        
        return chunk_facts

def get_facts_from_chunk(client, chunk, prompt, timestamp):
    """
    Extract facts from a single chunk using OpenAI API
    """
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # 准备系统提示，包含时间戳信息
            system_prompt = prompt.format(conversation_date=timestamp) if "conversation_date" in prompt else prompt
            
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content.strip()
            print(f"Raw response: {content[:]}...")  # Print full raw response for debugging
            
            # Remove code blocks if present (handle multiple variants)
            # Variant 1: ```json ... ```
            code_block_pattern = re.compile(r'```(?:json|JSON)?\s*([\s\S]*?)\s*```')
            match = code_block_pattern.search(content)
            if match:
                content = match.group(1).strip()
            
            # Variant 2: JSON within text paragraphs
            if not (content.startswith("{") and content.endswith("}")):
                # Try to extract JSON object from the content
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                else:
                    # Try to extract JSON array if no object found
                    json_array_match = re.search(r'\[[\s\S]*\]', content)
                    if json_array_match:
                        content = json_array_match.group(0)
                        # Wrap array in an object to match expected format
                        content = f'{{"facts": {content}}}'
                    else:
                        raise ValueError(f"No valid JSON found in response: {content[:100]}...")
            
            print(f"Extracted JSON: {content[:100]}...")  # Print extracted JSON for debugging
            
            # Try to fix common JSON issues, but be more cautious
            fixed_content = content
            
            # Only attempt fixes if the JSON is not already valid
            try:
                # Test if the content is already valid JSON
                json.loads(content)
                # If valid, skip fixing to avoid introducing errors
                print("JSON is already valid, skipping fixes")
                fixed_content = content
            except json.JSONDecodeError:
                print("JSON is invalid, attempting to fix...")
                
                # Fix 1: Only fix unquoted keys, but be cautious
                # This fix is only safe for simple key-value pairs without spaces
                fixed_content = re.sub(r'([{,\s])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_content)
                
                # Fix 2: Fix unquoted string values, but only for simple values
                # This fix is more conservative
                fixed_content = re.sub(r':\s*([a-zA-Z0-9_]+)\s*([,}\]])', r':"\1"\2', fixed_content)
                
                # Fix 3: Fix missing commas between objects/arrays
                fixed_content = re.sub(r'\}\s*\{', r'\},\{', fixed_content)
                fixed_content = re.sub(r'\]\s*\{', r'\],\{', fixed_content)
                fixed_content = re.sub(r'\}\s*\[', r'\},\[', fixed_content)
                fixed_content = re.sub(r'\]\s*\[', r'\],\[', fixed_content)
                
                # Fix 4: Ensure proper JSON structure
                open_braces = fixed_content.count('{')
                close_braces = fixed_content.count('}')
                open_brackets = fixed_content.count('[')
                close_brackets = fixed_content.count(']')
                
                # Add missing closing braces/brackets if needed
                while close_braces < open_braces:
                    fixed_content += '}'
                    close_braces += 1
                while close_brackets < open_brackets:
                    fixed_content += ']'
                    close_brackets += 1
            
            print(f"Fixed JSON: {fixed_content[:100]}...")  # Print first 100 chars of fixed JSON
            
            # Try to parse the JSON
            try:
                # Try to parse the fixed JSON
                facts_data = json.loads(fixed_content)
                
                # 保留完整的fact对象，包括details信息
                facts = []
                fact_objects = facts_data.get("facts", [])
                for fact_obj in fact_objects:
                    if fact_obj.get("fact"):
                        facts.append({
                            "fact": fact_obj.get("fact", ""),
                            "details": fact_obj.get("details", [])
                        })
                return facts
            except json.JSONDecodeError:
                # Try to use json5 if available
                try:
                    import json5
                    facts_data = json5.loads(content)
                    
                    # 保留完整的fact对象，包括details信息
                    facts = []
                    fact_objects = facts_data.get("facts", [])
                    for fact_obj in fact_objects:
                        if fact_obj.get("fact"):
                            facts.append({
                                "fact": fact_obj.get("fact", ""),
                                "details": fact_obj.get("details", [])
                            })
                    return facts
                except Exception:
                    # Try to extract just the facts array as a last resort
                    facts_match = re.search(r'"facts"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                    if facts_match:
                        facts_array_str = facts_match.group(1)
                        try:
                            facts_array = json.loads(f"[{facts_array_str}]")
                            
                            # 保留完整的fact对象，包括details信息
                            facts = []
                            for fact_obj in facts_array:
                                if fact_obj.get("fact"):
                                    facts.append({
                                        "fact": fact_obj.get("fact", ""),
                                        "details": fact_obj.get("details", [])
                                    })
                            return facts
                        except json.JSONDecodeError:
                            # As a last resort, return empty list
                            return []
                    else:
                        # No facts found, return empty list
                        return []
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON (Attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Raw API response: {content}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return []
        except Exception as e:
            print(f"Error extracting facts from chunk (Attempt {attempt + 1}/{max_retries}): {e}######## Raw response: {content[:]}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return []

def process_mem_alpha_dataset(input_file, output_file, client, prompt):
    """
    Process the mem-alpha-train.jsonl dataset and extract facts with timestamps
    Write results in real-time for immediate viewing
    Output format: {"id", "facts_of_chunks", "questions_and_answers", "data_source", "num_chunks", "num_questions"}
    """
    # Read existing progress if any
    progress = read_progress()
    print(f"Loaded progress: {len(progress)} instances already processed")
    
    # Load existing processed instances if output file exists
    processed_instances = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                processed_instances = json.load(f)
            print(f"Loaded {len(processed_instances)} instances from existing output file")
        except json.JSONDecodeError:
            print(f"Error parsing existing output file. Starting with empty results.")
            processed_instances = []
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting with empty results.")
            processed_instances = []
    
    # Read and process each line in the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)
    
    # Process each line with progress bar
    for line_num, line in tqdm(enumerate(lines, 1), total=total_lines, desc="Processing lines"):
        try:
            # Parse JSON line
            data = json.loads(line.strip())
            instance_id = data.get("instance_id")
            chunks = data.get("chunks", [])
            data_source = data.get("data_source")
            num_chunks = data.get("num_chunks")
            num_questions = data.get("num_questions")
            questions_and_answers = data.get("questions_and_answers", [])
            
            # Skip if instance already processed
            if instance_id in progress:
                print(f"Skipping instance {instance_id} (already processed)")
                continue
            
            # Initialize instance data
            instance_data = {
                "id": instance_id,
                "chunks": chunks,
                "facts_of_chunks": [],  # List to store each chunk's facts
                "questions_and_answers": questions_and_answers,  # Empty list as no Q&A in current data
                "data_source": data_source,
                "num_chunks": num_chunks,
                "num_questions": num_questions  # No questions in current data
            }
            
            # Process chunks in parallel
            instance_facts = []
            
            # Create a thread pool for this instance
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(process_single_chunk, chunk, client, prompt): chunk 
                    for chunk in chunks
                }
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_chunk), total=len(chunks),
                                 desc=f"Instance {instance_id} chunks", leave=False):
                    try:
                        chunk_facts = future.result()
                        # Only add chunk facts if it's not empty
                        if chunk_facts:
                            instance_facts.append(chunk_facts)
                    except Exception as e:
                        tqdm.write(f"Error processing chunk: {e}")
                        continue
            
            # Update instance data with processed facts
            instance_data["facts_of_chunks"] = instance_facts
            
            # Add processed instance to results
            processed_instances.append(instance_data)
            
            # Add instance to progress list
            progress.append(instance_id)
            
            # Write results to file in real-time with thread safety
            with OUTPUT_LOCK:
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_instances, out_f, ensure_ascii=False, indent=2)
                tqdm.write(f"✓ Saved {len(processed_instances)} instances to {output_file}")
            
            # Update progress file
            write_progress(progress)
            
        except json.JSONDecodeError as e:
            tqdm.write(f"Error parsing JSON at line {line_num}: {e}")
            continue
        except Exception as e:
            tqdm.write(f"Error processing line {line_num}: {e}")
            continue
    
    print(f"Processing complete! Processed {len(processed_instances)} instances from {total_lines} input lines.")
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = "/mnt/afs/codes/ljl/Memory-Agent/data/mem-alpha-timestamp.jsonl"
    output_file = "/mnt/afs/codes/ljl/Memory-Agent/data/mem-alpha-facts-QA-4.1-1229.json"
    
    # Process the dataset
    process_mem_alpha_dataset(input_file, output_file, client, MODIFIED_MEMREADER_PROMPT)
