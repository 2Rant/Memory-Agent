from copy import deepcopy
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct, PointIdsList
from utils import (
    get_embedding, parse_messages, FACT_RETRIEVAL_PROMPT, 
    remove_code_blocks, extract_json, get_update_memory_messages, 
    LME_JUDGE_MODEL_TEMPLATE, LME_ANSWER_PROMPT 
)
from lme_eval import lme_grader 
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import pytz
from datetime import datetime, timezone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
dimension=1536
collection_name = "lme_test"
vect_store_client = QdrantClient(path="./qdrant_db")
topk = 5
system_prompt = FACT_RETRIEVAL_PROMPT

if not os.path.exists("./qdrant_db/collection/" + collection_name):
    vect_store_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.DOT),
    )

def search(collection_name, vect_store_client, query_vector, top_k=5):
    search_result = vect_store_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        with_payload=True,
        limit=top_k
    ).points
    # print(search_result)
    return search_result

def insert(collection_name, vect_store_client, vectors, payloads=None):
    points = [
        PointStruct(id=idx, vector=vector, payload=payloads[idx])
        for idx, vector in enumerate(vectors)
    ]
    vect_store_client.upsert(
        collection_name=collection_name,
        points=points
    )

def generate_response(llm_client, question, question_date, context):
    prompt = LME_ANSWER_PROMPT.format(
        question=question,
        question_date=question_date,
        context=context
    )
    response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                # response_format={"type": "json_object"},
                temperature=0,
            )

    return response

with open("./data/longmemeval_s_cleaned.json", "r") as f:
    lines = json.load(f)[:1]

def process_user_memory(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    question = line.get("question")
    golden_answer = line.get("answer")

    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}

    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        parsed_messages = parse_messages(session) 
        # print("parsed_messages:", parsed_messages) 
        user_prompt = f"Input:\n{parsed_messages}"
        llm_response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )
        response = llm_response.choices[0].message.content
        # print(f"LLM 返回的原始响应: {response}")
        if response == '{"facts" : []}':
            # print("parsed_messages:", parsed_messages)
            pass
        try:
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    # First try direct JSON parsing
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    # Try extracting JSON from response using built-in function
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            print(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []
        # print(f"新检索到的事实: {new_retrieved_facts}") 

        if not new_retrieved_facts:
            # print("No new facts retrieved; skipping memory update.")
            continue
        retrieved_old_facts = []
        new_message_embeddings = {} 
        try:
            for fact in new_retrieved_facts:
                embedding_vector = get_embedding(openai_client, fact, dimension=dimension)
                new_message_embeddings[fact] = embedding_vector 
                
                existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
                # print(f"检索到的记忆点: {existing_memories}")
                for mem in existing_memories:
                    retrieved_old_facts.append({"id": mem.id, "text": mem.payload.get("data", "")})
                    # print("mem:", mem) 
            # print(f"检索到的旧事实: {retrieved_old_facts}")
        except Exception as e:
            print(f"生成嵌入时出错，请检查您的 API Key 和网络连接：{e}")

        unique_data = {}
        for item in retrieved_old_facts:
            unique_data[item["id"]] = item
        retrieved_old_facts = list(unique_data.values())

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_facts):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_facts[idx]["id"] = str(idx)
        # print(f"临时 UUID 映射: {temp_uuid_mapping}") 
        # print(f"用于记忆更新的旧事实: {retrieved_old_facts}") 

        if new_retrieved_facts:
            memory_action_prompt = get_update_memory_messages(retrieved_old_facts, new_retrieved_facts)
            # print("用于更新记忆的提示:", memory_action_prompt)
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": memory_action_prompt}],
                response_format={"type": "json_object"},
            )
            update_response = response.choices[0].message.content
            # print("update_response:", update_response)
            try:
                if not update_response.strip() or not update_response:
                    print("Empty response for memory update.")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(update_response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                print(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}

        else:
            new_memories_with_actions = {}

        returned_memories = []
        # print(f"new_memories_with_actions: {new_memories_with_actions}")
        try:
            for resp in new_memories_with_actions.get("memory", []):
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        print("Skipping memory entry because of empty `text` field.")
                        continue
                        
                    event_type = resp.get("event")
                    if event_type in operation_counts:
                        operation_counts[event_type] += 1
                        
                    embedding_vector = get_embedding(openai_client, action_text, dimension=dimension) # 原始代码在这里生成嵌入
                    
                    if event_type == "ADD":
                        memory_id = str(uuid.uuid4())
                        vect_store_client.upsert(
                        collection_name=collection_name, wait=True,
                        points=[ PointStruct(
                                    id=memory_id, vector=embedding_vector, 
                                    payload={ "data": action_text, "created_at": date_string.isoformat()}
                                ) ],
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type}) # (保留原始记录)
                    
                    elif event_type == "UPDATE":
                        points_data = vect_store_client.retrieve(
                            collection_name=collection_name,
                            ids=[temp_uuid_mapping.get(resp.get("id"))],
                            with_payload=True, 
                        )
                        result = points_data[0] if points_data else None
                        if result:
                            old_memory = result.payload.get("data", "")
                        else:
                            old_memory = ""

                        new_updated_at = date_string.isoformat()
                        result.payload["data"] = action_text
                        result.payload["updated_at"] = new_updated_at
                        vect_store_client.upsert(
                            collection_name=collection_name,
                            points=[ PointStruct(
                                        id=temp_uuid_mapping.get(resp.get("id")), 
                                        vector=embedding_vector, 
                                        payload=result.payload
                                    ) ],
                        )
                        returned_memories.append( 
                            {
                                "id": temp_uuid_mapping.get(resp.get("id"), "update_get_id_error"),
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": old_memory,
                            }
                        )
                    elif event_type == "DELETE":
                        if temp_uuid_mapping.get(resp.get("id")) is None:
                            print(f"Warning: Attempted DELETE on unknown temporary ID: {resp.get('id')}. Skipping.")
                            continue
                        # print(f"Deleting memory with ID: {temp_uuid_mapping.get(resp.get('id'))}")
                        vect_store_client.delete(
                            collection_name=collection_name,
                            points_selector=PointIdsList(
                                points=[temp_uuid_mapping.get(resp.get("id"))]
                            ),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping.get(resp.get("id"), "delete_get_id_error"),
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping.get(resp.get("id"), "none_get_id_error"),
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                except Exception as e:
                    print(f"Error processing memory action {resp}: {e}")       
        except Exception as e:
            print(f"Error iterating new_memories_with_actions: {e}")

        print(f"最终返回的记忆操作结果: {returned_memories}") 

    return operation_counts


def response_user(line):
    question = line.get("question")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    question = line.get("question")
    question_vector = get_embedding(openai_client, question, dimension=dimension)
    retrieved_memories = search(collection_name, vect_store_client, question_vector, top_k=topk)
    # context = "\n".join([mem.payload.get("data", "") for mem in retrieved_memories])
    memories_str = (
            "\n".join(
                f"- {mem.payload.get('created_at', '')}: {mem.payload.get('data', '')}"
                for mem in retrieved_memories
            )
        )
    response = generate_response(openai_client, question, question_date_string, memories_str)
    answer = response.choices[0].message.content

    return answer


evaluation_results = []
correct_count = 0
total_evaluated = 0
# **用于存储每个用户的详细统计信息**
user_detail_results = [] 
# **用于统计所有用户的总操作数**
total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}


with open("./data/longmemeval_s_cleaned.json", "r") as f:
    lines = json.load(f)[:2]

for idx, line in enumerate(lines):
    user_index = idx + 1
    print(f"\n\n==== 处理第 {user_index} 个用户的记忆 (存储阶段) ====")
    
    # **修改：捕获 process_user_memory 返回的计数**
    memory_counts = process_user_memory(line)
    
    print(f"\n\n==== 为第 {user_index} 个用户生成回答 (检索阶段) ====")
    answer = response_user(line)
    golden_answer = line.get("answer") # 获取黄金答案
    question = line.get("question") # 获取问题

    print(f"生成的回答: {answer}") # (保留原始打印)
    print(f"黄金答案: {golden_answer}") # (保留原始打印)
    
    # 2. 调用 Grader 进行评估
    is_correct = lme_grader(openai_client, question, golden_answer, answer)
    
    # 3. 统计结果
    total_evaluated += 1
    if is_correct:
        correct_count += 1
        evaluation_results.append(True)
    else:
        evaluation_results.append(False)

    print(f"LLM 评估结果: {'CORRECT' if is_correct else 'WRONG'}") # (保留原始打印)
    print(f"当前累计准确率: {correct_count / total_evaluated:.4f} ({correct_count}/{total_evaluated})") # (保留原始打印)

    # **新增：存储当前用户的详细结果**
    user_detail_results.append({
        "index": user_index,
        "is_correct": is_correct,
        "counts": memory_counts,
        "question": question
    })
    
    # **新增：累积总操作数**
    for key in total_memory_counts:
        total_memory_counts[key] += memory_counts.get(key, 0)


# 4. 计算最终总准确率
print("\n\n==================================================")
print("             🎯 最终评估结果") 
print("==================================================")

if total_evaluated > 0:
    final_accuracy = correct_count / total_evaluated
    print(f"总评估问题数: {total_evaluated}")
    print(f"正确回答数: {correct_count}")
    print(f"最终总准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
else:
    print("没有评估任何问题。")
print("==================================================")

# **5. 新增：打印每个用户的详细操作统计**
print("\n\n==================================================")
print("        📊 详细记忆操作统计 (按用户)")
print("==================================================")

for res in user_detail_results:
    user_index = res["index"]
    is_correct = res["is_correct"]
    counts = res["counts"]
    question = res["question"]
    
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    
    print(f"\n--- 用户/问题 {user_index} ---")
    print(f"  问题: {question[:60]}...")
    print(f"  评估结果: {status}")
    print(f"  记忆操作: ADD={counts.get('ADD', 0)}, UPDATE={counts.get('UPDATE', 0)}, DELETE={counts.get('DELETE', 0)}, NONE={counts.get('NONE', 0)}")

print("\n--- 所有用户的记忆操作总览 ---")
print(f"  ADD (新增):    {total_memory_counts['ADD']}")
print(f"  UPDATE (更新): {total_memory_counts['UPDATE']}")
print(f"  DELETE (删除): {total_memory_counts['DELETE']}")
print(f"  NONE (无操作): {total_memory_counts['NONE']}")
print("==================================================")