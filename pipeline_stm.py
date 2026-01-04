from copy import deepcopy
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct, PointIdsList
from utils import (
    get_embedding, parse_messages, 
    remove_code_blocks, extract_json, 
    LME_JUDGE_MODEL_TEMPLATE, LME_ANSWER_PROMPT 
)
from lme_eval import lme_grader 
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import pytz
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL, timeout=60.0, max_retries=3)
dimension = 1536
collection_name = "_lme"

vect_store_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60.0
)
topk = 5
search_topk = 10

# =============================================================================
# Prompts for Short Term Memory (STM)
# =============================================================================

STM_UPDATE_PROMPT = """You are a conversation summarizer. Maintain a rolling summary (Short Term Memory) of the conversation.

**Current Summary:**
{current_stm}

**New Dialogue:**
{new_dialogue}

**Instructions:**
1. Update the summary to include key context from the new dialogue (topics, entities, constraints).
2. Keep it concise (max 3 sentences).
3. Discard irrelevant details from long ago.
4. Output ONLY the updated summary text.
"""

FACT_RETRIEVAL_STM_TEMPLATE = """You are a Personal Information Organizer. Extract relevant facts.

Here are some few shot examples:
Input: Hi.
Output: {{"facts" : []}}
Input: I love apples.
Output: {{"facts" : ["User loves apples"]}}

Return the facts in a json format: {{"facts": ["fact1", "fact2"]}}

Remember:
- Today's date is {current_date}.
- **Recent Context (Short Term Memory)**: {short_term_memory}
- Use the context to resolve ambiguity (e.g. "it", "he") but extract facts based on the dialogue below.

Following is the conversation:
"""

# Update Decision Prompt (Standard)
DEFAULT_UPDATE_MEMORY_PROMPT = """You are a smart memory manager.
Compare newly retrieved facts with existing memory.
Decide: ADD, UPDATE, DELETE, or NONE.

Guidelines:
1. ADD: New info.
2. UPDATE: New details on existing topic.
3. DELETE: Contradictory info.
4. NONE: Redundant info.

Return JSON only:
{{
    "memory" : [
        {{
            "id" : "<ID>",
            "text" : "<Content>",
            "event" : "<ADD/UPDATE/DELETE/NONE>",
            "old_memory" : "<Old content>"
        }},
        ...
    ]
}}
"""

def get_update_memory_messages_stm(retrieved_old_memory_dict, response_content, stm_context="", custom_update_memory_prompt=None):
    if custom_update_memory_prompt is None:
        custom_update_memory_prompt = DEFAULT_UPDATE_MEMORY_PROMPT

    stm_part = f"""
    **Conversation Context (Short Term Memory):**
    ```
    {stm_context if stm_context else "No context yet."}
    ```
    """

    if retrieved_old_memory_dict:
        current_memory_part = f"""
    **Existing Vector Memory:**
    ```
    {retrieved_old_memory_dict}
    ```
    """
    else:
        current_memory_part = "Existing Vector Memory is empty."

    return f"""{custom_update_memory_prompt}

    {stm_part}
    {current_memory_part}

    **New Facts:**
    ```
    {response_content}
    ```
    
    Do not return anything except the JSON format.
    """

# =============================================================================

def search(collection_name, vect_store_client, query_vector, top_k=5):
    search_result = vect_store_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        with_payload=True,
        limit=top_k
    ).points
    return search_result

def generate_response(llm_client, question, question_date, context, stm=""):
    # [注入] 回答时带上 STM
    full_context = f"Recent Context (STM):\n{stm}\n\nRetrieved Facts:\n{context}"
    
    prompt = LME_ANSWER_PROMPT.format(
        question=question,
        question_date=question_date,
        context=full_context
    )
    response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
    return response

# =============================================================================
# Logic 1: Infer Mode (STM + Intelligent Update)
# =============================================================================
def process_user_memory_infer(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    
    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    current_stm = "Start of conversation." # 初始化 STM

    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        # 逐轮处理 (Per Turn) 以便 STM 能逐轮滚动更新
        # 如果你想整段 Session 处理，也可以把这个 for 循环去掉，直接传 session
        # 但 STM 的精髓在于由前推后，逐轮效果最好
        for turn_id in range(0, len(session), 2):
            # 获取当前轮对话 (User + Assistant)
            current_turn_msgs = session[turn_id:turn_id+2]
            parsed_messages = parse_messages(current_turn_msgs)

            # --- Step 1: 更新 STM ---
            try:
                stm_prompt = STM_UPDATE_PROMPT.format(
                    current_stm=current_stm,
                    new_dialogue=parsed_messages
                )
                stm_response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": stm_prompt}], 
                    temperature=0, 
                )
                updated_stm = stm_response.choices[0].message.content.strip()
                if updated_stm:
                    current_stm = updated_stm
                    # print(f"--> Updated STM: {current_stm}")
            except Exception as e:
                print(f"Error updating STM: {e}")

            # --- Step 2: 事实抽取 (注入 STM) ---
            current_system_prompt = FACT_RETRIEVAL_STM_TEMPLATE.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                short_term_memory=current_stm
            )
            
            user_prompt = f"Input:\n{parsed_messages}"
            llm_response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": current_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
            )
            response = llm_response.choices[0].message.content
            
            # 解析 Facts
            new_retrieved_facts = []
            try:
                response_str = remove_code_blocks(response)
                if response_str.strip():
                    parsed = json.loads(response_str)
                    if isinstance(parsed, dict): new_retrieved_facts = parsed.get("facts", [])
                    elif isinstance(parsed, list): new_retrieved_facts = parsed
            except Exception as e:
                print(f"Error parsing facts: {e}")

            if not new_retrieved_facts: continue
            
            # --- 准备旧事实 ---
            retrieved_old_facts = []
            try:
                for fact in new_retrieved_facts:
                    if isinstance(fact, dict): fact_str = json.dumps(fact, ensure_ascii=False)
                    else: fact_str = str(fact)
                    if not fact_str.strip(): continue

                    embedding_vector = get_embedding(openai_client, fact_str, dimension=dimension)
                    existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
                    for mem in existing_memories:
                        retrieved_old_facts.append({"id": mem.id, "text": mem.payload.get("data", "")})
            except Exception as e:
                print(f"Error searching old facts: {e}")

            unique_data = {}
            for item in retrieved_old_facts: unique_data[item["id"]] = item
            retrieved_old_facts = list(unique_data.values())

            temp_uuid_mapping = {}
            for idx, item in enumerate(retrieved_old_facts):
                temp_uuid_mapping[str(idx)] = item["id"]
                retrieved_old_facts[idx]["id"] = str(idx)
            
            # --- Step 3: 决策与更新 ---
            new_memories_with_actions = {}
            if new_retrieved_facts:
                facts_str_display = json.dumps(new_retrieved_facts, ensure_ascii=False, indent=2)
                
                # 传入 STM 作为 Context
                memory_action_prompt = get_update_memory_messages_stm(
                    retrieved_old_memory_dict=retrieved_old_facts, 
                    response_content=facts_str_display,
                    stm_context=current_stm 
                )
                
                try:
                    response = openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": memory_action_prompt}],
                        response_format={"type": "json_object"},
                    )
                    update_response = remove_code_blocks(response.choices[0].message.content)
                    if update_response.strip():
                        new_memories_with_actions = json.loads(update_response)
                except Exception as e:
                    print(f"Error in update decision: {e}")

            # --- Step 4: 执行 Qdrant 操作 ---
            try:
                for resp in new_memories_with_actions.get("memory", []):
                    action_text = resp.get("text")
                    if not action_text: continue
                    
                    if isinstance(action_text, dict): action_text_str = json.dumps(action_text, ensure_ascii=False)
                    else: action_text_str = str(action_text)

                    if temp_uuid_mapping.get(resp.get("id")) is None:
                        for item in retrieved_old_facts:
                            if item["text"] == resp.get("old_memory"): resp["id"] = item["id"]
                    
                    event_type = resp.get("event")
                    if event_type in operation_counts: operation_counts[event_type] += 1
                    
                    embedding_vector = get_embedding(openai_client, action_text_str, dimension=dimension)
                    
                    if event_type == "ADD":
                        memory_id = str(uuid.uuid4())
                        vect_store_client.upsert(collection_name=collection_name, wait=True,
                            points=[PointStruct(id=memory_id, vector=embedding_vector, payload={"data": action_text_str, "created_at": date_string.isoformat()})])
                    elif event_type == "UPDATE":
                        real_id = temp_uuid_mapping.get(resp.get("id"))
                        if real_id:
                            points_data = vect_store_client.retrieve(collection_name=collection_name, ids=[real_id], with_payload=True)
                            if points_data:
                                payload = points_data[0].payload
                                payload["data"] = action_text_str
                                payload["updated_at"] = date_string.isoformat()
                                vect_store_client.upsert(collection_name=collection_name, wait=True, points=[PointStruct(id=real_id, vector=embedding_vector, payload=payload)])
                    elif event_type == "DELETE":
                        real_id = temp_uuid_mapping.get(resp.get("id"))
                        if real_id:
                            vect_store_client.delete(collection_name=collection_name, wait=True, points_selector=PointIdsList(points=[real_id]))
            except Exception as e:
                print(f"Error executing ops: {e}")

    return operation_counts, current_stm

# =============================================================================
# Logic 2: Non-Infer Mode (STM + Direct ADD)
# =============================================================================
def process_user_memory(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    current_stm = "Start of conversation." 
    
    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        # 逐轮处理
        for turn_id in range(0, len(session), 2):
            parsed_messages = parse_messages(session[turn_id:turn_id+2])

            # [Step 1] 维护 STM
            try:
                stm_prompt = STM_UPDATE_PROMPT.format(
                    current_stm=current_stm,
                    new_dialogue=parsed_messages
                )
                stm_response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": stm_prompt}], 
                    temperature=0, 
                )
                updated_stm = stm_response.choices[0].message.content.strip()
                if updated_stm: current_stm = updated_stm
            except Exception as e:
                pass

            # [Step 2] 事实抽取
            current_system_prompt = FACT_RETRIEVAL_STM_TEMPLATE.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                short_term_memory=current_stm
            )
            user_prompt = f"Input:\n{parsed_messages}"
            
            try:
                llm_response = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                response_str = remove_code_blocks(llm_response.choices[0].message.content)
                
                new_retrieved_facts = []
                if response_str.strip():
                    parsed = json.loads(response_str)
                    if isinstance(parsed, dict): new_retrieved_facts = parsed.get("facts", [])
                    elif isinstance(parsed, list): new_retrieved_facts = parsed
            except Exception as e:
                new_retrieved_facts = []

            # [Step 3] 直接 ADD
            for fact in new_retrieved_facts:
                if isinstance(fact, dict): fact_str = json.dumps(fact, ensure_ascii=False)
                else: fact_str = str(fact)
                if not fact_str.strip(): continue

                try:
                    embedding_vector = get_embedding(openai_client, fact_str, dimension=dimension)
                    memory_id = str(uuid.uuid4())
                    vect_store_client.upsert(
                        collection_name=collection_name, 
                        wait=True,
                        points=[PointStruct(id=memory_id, vector=embedding_vector, payload={"data": fact_str, "created_at": date_string.isoformat()})]
                    )
                    operation_counts["ADD"] += 1
                except Exception as e:
                    print(f"Error adding fact: {e}")
        
    return operation_counts, current_stm


def response_user(line, stm):
    question = line.get("question")
    question_date = line.get("question_date") + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    
    question_vector = get_embedding(openai_client, question, dimension=dimension)
    retrieved_memories = search(collection_name, vect_store_client, question_vector, top_k=search_topk)
    
    memories_str = "\n".join(f"- {mem.payload.get('created_at', '')}: {mem.payload.get('data', '')}" for mem in retrieved_memories)
    
    response = generate_response(openai_client, question, question_date_string, memories_str, stm)
    answer = response.choices[0].message.content
    return answer

def process_and_evaluate_user(line, user_index, client, infer):
    try:
        if infer:
            memory_counts, stm = process_user_memory_infer(line)
        else:
            memory_counts, stm = process_user_memory(line)
        
        answer = response_user(line, stm)
        golden_answer = line.get("answer") 
        question = line.get("question")
        is_correct = lme_grader(client, question, golden_answer, answer)
        
        return {
            "index": user_index, "is_correct": is_correct, "counts": memory_counts,
            "stm": stm, "question": question, "answer": answer, "golden_answer": golden_answer
        }
    except Exception as e:
        print(f"Error processing user {user_index}: {e}")
        return {"index": user_index, "is_correct": False, "counts": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}, "question": line.get("question", "N/A")}

if __name__ == "__main__":
    try:
        if vect_store_client.collection_exists(collection_name=collection_name):
            vect_store_client.delete_collection(collection_name=collection_name)
        vect_store_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.DOT),
        )
    except Exception as e:
        print(f"Qdrant Init Error: {e}")
        exit()

    with open("./data/longmemeval_s_cleaned.json", "r") as f:
        lines = json.load(f)[:50] 
    
    print(f"已加载 {len(lines)} 个用户/问题。")
    total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    
    MAX_WORKERS = 1 
    infer = True 

    print(f"开始使用 {MAX_WORKERS} 个线程处理 (infer={infer})...")
    user_detail_results = [] 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_and_evaluate_user, line, idx + 1, openai_client, infer=infer) for idx, line in enumerate(lines)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Progress"):
            user_detail_results.append(future.result())

    user_detail_results.sort(key=lambda x: x.get("index", 0))
    correct_count = 0
    for res in user_detail_results:
        if res.get("is_correct"): correct_count += 1
        for k in total_memory_counts: total_memory_counts[k] += res.get("counts", {}).get(k, 0)

    print("\n=== Final Results ===")
    print(f"Accuracy: {correct_count}/{len(user_detail_results)} ({(correct_count/len(user_detail_results))*100:.2f}%)")
    
    for res in user_detail_results:
        print(f"\nUser {res['index']}: {res['question']}")
        print(f"STM (Final): {res.get('stm', 'N/A')}")
        print(f"Result: {'✅' if res['is_correct'] else '❌'}")
    
    print(f"\nTotal Ops: {total_memory_counts}")