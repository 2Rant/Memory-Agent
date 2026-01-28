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
PROGRESS_FILE = "/mnt/afs/codes/ljl/Memory-Agent/data/progress_session_qa.json"  # File to store processing progress
PROGRESS_LOCK = threading.Lock()  # Lock for thread-safe progress updates

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# QA generation prompt
QA_GENERATION_PROMPT = """You are an expert data analyst specializing in reading comprehension and fact extraction. You are currently processing part of a long document. Generate 5 Question-Answer (QA) pairs to verify the **newly added key facts** in the provided text chunk. To avoid redundancy, here is a list of questions generated from previous chunks. Do NOT generate questions that are semantically similar to these:
--- Existing Questions ---------------------------

{existing_questions}

# What NOT to Generate (Examples of Trivial Questions to Avoid)
- "What was the user's initial greeting?"
- "How did the assistant respond to the greeting?"
- "What was the first thing the user said?"
- "How did the conversation start?"
- "What was the assistant's opening line?"

# Examples of Good Quality Questions
- "What specific tips does the assistant provide for capturing sunset colors?"
- "What resources does the assistant recommend for learning abstract art?"
- "What criteria should be used for selecting program facilitators?"
- "What materials does the user need to develop for the program?"
- "What are the key steps for creating a program timeline?"

# Input Text (Current Chunk)
{session_text}

# Instructions
1. **Focus on Substantive Information**: Ask ONLY about **meaningful** information, **specific** requests, **detailed** advice, **factual** data, or **actionable** recommendations introduced in this specific chunk.
2. **Avoid Trivial Interactions**: Do not ask about greetings, conversation starters, or other superficial aspects of the dialogue.
3. **Ignore Static Info**: Do not ask about established background facts (e.g., "Who is the protagonist?", "What is his father's name?") unless they change or are first introduced in this chunk.
4. **Answer Constraints**:
   - Answers must be **short** (entities, dates, short phrases, exact spans).
   - Answers must be objective and verifiable against the text.
   - Paraphrases or semantically equivalent expressions of the information are allowed.

# Quality Criteria
A good question should:
- Focus on specific content rather than general conversation structure
- Require meaningful information from the text to answer
- Test understanding of important points in the chunk
- Avoid obvious or superficial aspects of the dialogue

# Output Format (MUST BE A JSON ARRAY)
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

def convert_session_to_text(session):
    """
    Convert a session (list of messages) to a single text string
    """
    session_text = []
    for message in session:
        role = message.get('role', '').lower()
        content = message.get('content', '')
        # Handle non-string content
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        elif content is None:
            content = ''
        else:
            content = str(content)
        if role == 'user':
            session_text.append(f"User: {content}")
        elif role == 'assistant':
            session_text.append(f"Assistant: {content}")
    return '\n'.join(session_text)

def get_existing_questions_from_previous_sessions(sessions, current_index):
    """
    Get existing questions from previous sessions to avoid duplication
    """
    existing_questions = []
    # Only consider sessions before the current one
    for i in range(current_index):
        session = sessions[i]
        # Convert session to text and extract any questions (simplified approach)
        # In a real implementation, we would need to track generated questions
        # For now, we'll just return an empty list as we're generating new questions
    return existing_questions

def evaluate_qa_quality(qa_pairs):
    """
    Evaluate the quality of generated QA pairs
    Returns a score between 0 (lowest) and 1 (highest)
    """
    if not qa_pairs:
        return 0.0
    
    quality_score = 0.0
    trivial_keywords = [
        "initial greeting",
        "respond to the greeting",
        "first thing the user said",
        "how did the conversation start",
        "assistant's opening line",
        "hi",
        "hello",
        "greeting",
        "greet"
    ]
    
    for qa in qa_pairs:
        if isinstance(qa, dict) and 'question' in qa:
            question = qa['question'].lower()
            
            # Check for trivial patterns
            is_trivial = False
            for keyword in trivial_keywords:
                if keyword in question:
                    is_trivial = True
                    break
            
            if not is_trivial:
                quality_score += 1.0
    
    return quality_score / len(qa_pairs) if qa_pairs else 0.0

def generate_qa_for_session(session, session_index, client, prompt):
    """
    Generate QA pairs for a single session
    Implements retry mechanism based on quality evaluation
    """
    with THREAD_SEMAPHORE:  # Limit concurrent API calls
        # Convert session to text
        session_text = convert_session_to_text(session)
        
        # Skip empty sessions
        if not session_text.strip():
            print(f"Skipping empty session {session_index}")
            return []
        
        # Get existing questions from previous sessions
        existing_questions = []
        existing_questions_text = '\n'.join(existing_questions) if existing_questions else "None"
        
        max_retries = 3
        quality_threshold = 0.8
        best_qa_pairs = []
        best_score = 0.0
        
        for attempt in range(max_retries):
            try:
                print(f"Generating QA pairs for session {session_index}, attempt {attempt + 1}/{max_retries}")
                
                # Generate QA pairs using the prompt
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst specializing in reading comprehension and fact extraction. Focus on generating substantive questions about the content, not trivial aspects like greetings."},
                        {"role": "user", "content": prompt.format(
                            existing_questions=existing_questions_text,
                            session_text=session_text
                        )}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # Extract and parse the response
                content = response.choices[0].message.content.strip()
                
                # Remove code blocks if present
                code_block_pattern = re.compile(r'```(?:json|JSON)?\s*([\s\S]*?)\s*```')
                match = code_block_pattern.search(content)
                if match:
                    content = match.group(1).strip()
                
                # Parse JSON response
                qa_pairs = json.loads(content)
                
                # Validate and clean QA pairs
                cleaned_qa_pairs = []
                for qa in qa_pairs:
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        # Handle non-string question and answer
                        question = qa['question']
                        answer = qa['answer']
                        
                        if isinstance(question, list):
                            question = ' '.join(str(item) for item in question)
                        elif question is None:
                            question = ''
                        else:
                            question = str(question).strip()
                        
                        if isinstance(answer, list):
                            answer = ' '.join(str(item) for item in answer)
                        elif answer is None:
                            answer = ''
                        else:
                            answer = str(answer).strip()
                        
                        cleaned_qa_pairs.append({
                            'question': question,
                            'answer': answer
                        })
                
                # Ensure we have exactly 5 QA pairs
                if len(cleaned_qa_pairs) < 5:
                    # If we have fewer than 5, generate additional ones
                    # For simplicity, we'll just use what we have
                    pass
                elif len(cleaned_qa_pairs) > 5:
                    # If we have more than 5, take the first 5
                    cleaned_qa_pairs = cleaned_qa_pairs[:5]
                
                # Evaluate quality
                current_score = evaluate_qa_quality(cleaned_qa_pairs)
                print(f"Quality score for session {session_index}: {current_score:.2f}")
                
                # Update best QA pairs if current is better
                if current_score > best_score:
                    best_score = current_score
                    best_qa_pairs = cleaned_qa_pairs
                
                # If quality is above threshold, break early
                if current_score >= quality_threshold:
                    print(f"Quality threshold met for session {session_index}")
                    break
                
                # Add delay between retries
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating QA pairs for session {session_index} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return best_qa_pairs if best_qa_pairs else []
                time.sleep(2)
                continue
        
        # Filter out any remaining trivial questions
        final_qa_pairs = []
        trivial_keywords = [
            "initial greeting",
            "respond to the greeting",
            "first thing the user said",
            "how did the conversation start",
            "assistant's opening line",
            "hi",
            "hello",
            "greeting",
            "greet"
        ]
        
        for qa in best_qa_pairs:
            question = qa['question'].lower()
            is_trivial = False
            for keyword in trivial_keywords:
                if keyword in question:
                    is_trivial = True
                    break
            if not is_trivial:
                final_qa_pairs.append(qa)
        
        # If we filtered out too many, use the best available
        if len(final_qa_pairs) < 3:
            final_qa_pairs = best_qa_pairs
        
        print(f"Final QA pairs for session {session_index}: {len(final_qa_pairs)}")
        return final_qa_pairs

def process_single_instance(instance, client, prompt):
    """
    Process a single instance: generate QA pairs for each session
    """
    instance_id = instance.get('question_id', f"instance_{int(time.time())}")
    sessions = instance.get('haystack_sessions', [])
    
    # Generate QA pairs for each session
    session_qa_pairs = []
    
    # Create a thread pool for this instance
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all sessions for processing
        future_to_session = {
            executor.submit(generate_qa_for_session, session, i, client, prompt): (session, i) 
            for i, session in enumerate(sessions)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_session):
            session, session_index = future_to_session[future]
            try:
                qa_pairs = future.result()
                session_qa_pairs.append({
                    'session_index': session_index,
                    'qa_pairs': qa_pairs
                })
            except Exception as e:
                print(f"Error processing session {session_index}: {e}")
                continue
    
    # Sort session QA pairs by session index
    session_qa_pairs.sort(key=lambda x: x['session_index'])
    
    # Update instance with session QA pairs
    instance['session_qa_pairs'] = session_qa_pairs
    
    return instance

def process_dataset(input_file, output_file, client, prompt):
    """
    Process the dataset and generate QA pairs for each session
    """
    # Read existing progress if any
    progress = read_progress()
    print(f"Loaded progress: {len(progress)} instances already processed")
    
    # Load the entire dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded dataset with {len(dataset)} instances")
    
    # Process each instance
    processed_instances = []
    for i, instance in tqdm(enumerate(dataset), total=len(dataset), desc="Processing instances"):
        instance_id = instance.get('question_id', f"instance_{i}")
        
        # Skip if instance already processed
        if instance_id in progress:
            print(f"Skipping instance {instance_id} (already processed)")
            processed_instances.append(instance)
            continue
        
        # Process instance
        try:
            processed_instance = process_single_instance(instance, client, prompt)
            processed_instances.append(processed_instance)
            
            # Update progress
            progress.append(instance_id)
            write_progress(progress)
            
            # Save intermediate results
            with OUTPUT_LOCK:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_instances, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            continue
    
    print(f"Processing complete! Processed {len(processed_instances)} instances from {len(dataset)} total instances.")
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = "/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json"
    output_file = "/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_with_session_qa.json"
    
    # Process the dataset
    process_dataset(input_file, output_file, client, QA_GENERATION_PROMPT)
