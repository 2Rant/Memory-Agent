# from copy import deepcopy
# import uuid
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from qdrant_client.models import PointStruct, PointIdsList
# from utils import (
#     get_embedding, parse_messages, FACT_RETRIEVAL_PROMPT, 
#     remove_code_blocks, extract_json, get_update_memory_messages, 
#     LME_JUDGE_MODEL_TEMPLATE, LME_ANSWER_PROMPT 
# )
# from lme_eval import lme_grader 
# from dotenv import load_dotenv
# import os
# import json
# from openai import OpenAI
# import pytz
# from datetime import datetime, timezone
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# BASE_URL = os.getenv("OPENAI_BASE_URL")
# MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
# dimension=1536
# collection_name = "lme"
# # vect_store_client = QdrantClient(path="./qdrant_db")
# vect_store_client = QdrantClient(url="https://bb9a565c-40a3-471b-917b-4fbaeb99446f.us-east4-0.gcp.cloud.qdrant.io:6333",
#                                   api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.07poGFmT_he2JW-a_Qy4CThgHpCG-uSC4le2VOlPTUo")
# topk = 5
# system_prompt = FACT_RETRIEVAL_PROMPT


# def search(collection_name, vect_store_client, query_vector, top_k=5):
#     search_result = vect_store_client.query_points(
#         collection_name=collection_name,
#         query=query_vector,
#         with_payload=True,
#         limit=top_k
#     ).points
#     # print(search_result)
#     return search_result

# def insert(collection_name, vect_store_client, vectors, payloads=None):
#     points = [
#         PointStruct(id=idx, vector=vector, payload=payloads[idx])
#         for idx, vector in enumerate(vectors)
#     ]
#     vect_store_client.upsert(
#         collection_name=collection_name,
#         points=points
#     )

# def generate_response(llm_client, question, question_date, context):
#     prompt = LME_ANSWER_PROMPT.format(
#         question=question,
#         question_date=question_date,
#         context=context
#     )
#     response = llm_client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[{"role": "system", "content": prompt}],
#                 # response_format={"type": "json_object"},
#                 temperature=0,
#             )

#     return response
 
# def process_user_memory_infer(line):
#     dates = line.get("haystack_dates")
#     sessions = line.get("haystack_sessions")
#     question_date = line.get("question_date")
#     question_date = question_date + " UTC"
#     question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
#     question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
#     question = line.get("question")
#     golden_answer = line.get("answer") 

#     operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}

#     for session_id, session in enumerate(sessions):
#         date = dates[session_id] + " UTC"
#         date_format = "%Y/%m/%d (%a) %H:%M UTC"
#         date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
#         parsed_messages = parse_messages(session) 
#         # print("parsed_messages:", parsed_messages) 
#         user_prompt = f"Input:\n{parsed_messages}"
#         llm_response = openai_client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format={"type": "json_object"},
#         )
#         response = llm_response.choices[0].message.content
#         # print(f"LLM 返回的原始响应: {response}")
#         if response == '{"facts" : []}':
#             # print("parsed_messages:", parsed_messages)
#             pass
#         try:
#             response = remove_code_blocks(response)
#             if not response.strip():
#                 new_retrieved_facts = []
#             else:
#                 try:
#                     new_retrieved_facts = json.loads(response)["facts"]
#                 except json.JSONDecodeError:
#                     extracted_json = extract_json(response)
#                     new_retrieved_facts = json.loads(extracted_json)["facts"]
#         except Exception as e:
#             print(f"Error in new_retrieved_facts: {e}")
#             new_retrieved_facts = []
#         # print(f"新检索到的事实: {new_retrieved_facts}") 

#         if not new_retrieved_facts:
#             # print("No new facts retrieved; skipping memory update.")
#             continue
            
#         retrieved_old_facts = []
#         new_message_embeddings = {} 
#         try:
#             for fact in new_retrieved_facts:
#                 embedding_vector = get_embedding(openai_client, fact, dimension=dimension)
#                 new_message_embeddings[fact] = embedding_vector 
                
#                 existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
#                 for mem in existing_memories:
#                     retrieved_old_facts.append({"id": mem.id, "text": mem.payload.get("data", "")})
#                     # print("mem:", mem) 
#         except Exception as e:
#             print(f"生成嵌入时出错，请检查您的 API Key 和网络连接：{e}")

#         unique_data = {}
#         for item in retrieved_old_facts:
#             unique_data[item["id"]] = item
#         retrieved_old_facts = list(unique_data.values())

#         temp_uuid_mapping = {}
#         for idx, item in enumerate(retrieved_old_facts):
#             temp_uuid_mapping[str(idx)] = item["id"]
#             retrieved_old_facts[idx]["id"] = str(idx)
#         # print(f"临时 UUID 映射: {temp_uuid_mapping}") 
#         # print(f"用于记忆更新的旧事实: {retrieved_old_facts}") 

#         if new_retrieved_facts:
#             memory_action_prompt = get_update_memory_messages(retrieved_old_facts, new_retrieved_facts)
#             response = openai_client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=[{"role": "user", "content": memory_action_prompt}],
#                 response_format={"type": "json_object"},
#             )
#             update_response = response.choices[0].message.content
#             # print("update_response:", update_response)
#             try:
#                 if not update_response.strip() or not update_response:
#                     # print("Empty response for memory update.")
#                     new_memories_with_actions = {}
#                 else:
#                     response = remove_code_blocks(update_response)
#                     new_memories_with_actions = json.loads(response)
#             except Exception as e:
#                 print(f"Invalid JSON response: {e}")
#                 new_memories_with_actions = {}

#         else:
#             new_memories_with_actions = {}

#         returned_memories = []
#         # print(f"new_memories_with_actions: {new_memories_with_actions}")
#         try:
#             for resp in new_memories_with_actions.get("memory", []):
#                 try:
#                     action_text = resp.get("text")
#                     if not action_text:
#                         print("Skipping memory entry because of empty `text` field.")
#                         continue
                        
#                     event_type = resp.get("event")
#                     if event_type in operation_counts:
#                         operation_counts[event_type] += 1
                        
#                     embedding_vector = get_embedding(openai_client, action_text, dimension=dimension)
                    
#                     if event_type == "ADD":
#                         memory_id = str(uuid.uuid4())
#                         vect_store_client.upsert(
#                         collection_name=collection_name, wait=True,
#                         points=[ PointStruct(
#                                     id=memory_id, vector=embedding_vector, 
#                                     payload={ "data": action_text, "created_at": date_string.isoformat()}
#                                 ) ],
#                         )
#                         returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type}) 
                    
#                     elif event_type == "UPDATE":
#                         points_data = vect_store_client.retrieve(
#                             collection_name=collection_name,
#                             ids=[temp_uuid_mapping.get(resp.get("id"))],
#                             with_payload=True, 
#                         )
#                         result = points_data[0] if points_data else None
#                         if result:
#                             old_memory = result.payload.get("data", "")
#                         else:
#                             old_memory = ""

#                         new_updated_at = date_string.isoformat()
#                         result.payload["data"] = action_text
#                         result.payload["updated_at"] = new_updated_at
#                         vect_store_client.upsert(
#                             collection_name=collection_name,
#                             points=[ PointStruct(
#                                         id=temp_uuid_mapping.get(resp.get("id")), 
#                                         vector=embedding_vector, 
#                                         payload=result.payload
#                                     ) ],
#                         )
#                         returned_memories.append( 
#                             {
#                                 "id": temp_uuid_mapping.get(resp.get("id"), "update_get_id_error"),
#                                 "memory": action_text,
#                                 "event": event_type,
#                                 "previous_memory": old_memory,
#                             }
#                         )
#                     elif event_type == "DELETE":
#                         if temp_uuid_mapping.get(resp.get("id")) is None:
#                             print(f"Warning: Attempted DELETE on unknown temporary ID: {resp.get('id')}. Skipping.")
#                             continue
#                         # print(f"Deleting memory with ID: {temp_uuid_mapping.get(resp.get('id'))}")
#                         vect_store_client.delete(
#                             collection_name=collection_name,
#                             points_selector=PointIdsList(
#                                 points=[temp_uuid_mapping.get(resp.get("id"))]
#                             ),
#                         )
#                         returned_memories.append(
#                             {
#                                 "id": temp_uuid_mapping.get(resp.get("id"), "delete_get_id_error"),
#                                 "memory": action_text,
#                                 "event": event_type,
#                             }
#                         )
#                     elif event_type == "NONE":
#                         returned_memories.append(
#                             {
#                                 "id": temp_uuid_mapping.get(resp.get("id"), "none_get_id_error"),
#                                 "memory": action_text,
#                                 "event": event_type,
#                             }
#                         )
#                 except Exception as e:
#                     print(f"Error processing memory action {resp}: {e}")       
#         except Exception as e:
#             print(f"Error iterating new_memories_with_actions: {e}")

#         print(f"最终返回的记忆操作结果: {returned_memories}") 

#     return operation_counts


# def process_user_memory(line):
#     dates = line.get("haystack_dates")
#     sessions = line.get("haystack_sessions")
#     returned_memories = []
#     operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
#     for session_id, session in enumerate(sessions):
#         date = dates[session_id] + " UTC"
#         date_format = "%Y/%m/%d (%a) %H:%M UTC"
#         date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)

#         # for message in session:
#         #     message_dict = message
#         #     if isinstance(message, str):
#         #         try:
#         #             message_dict = json.loads(message)
#         #         except json.JSONDecodeError:
#         #             print(f"无法解析消息为 JSON: {message}")
#         #             continue

#         #     metadata = {
#         #         "created_at": date_string.isoformat(),
#         #     }

#         #     memory_id = str(uuid.uuid4())
#         #     if message_dict["role"] == "system":
#         #         continue
            
#         #     metadata["role"] = message_dict["role"]
#         #     metadata["data"] = message_dict["content"]

#         #     msg_content = message_dict["content"]
#         #     embedding_vector = get_embedding(openai_client, msg_content, dimension=dimension)
#         #     vect_store_client.upsert(
#         #                 collection_name=collection_name, wait=True,
#         #                 points=[ PointStruct(
#         #                             id=memory_id, vector=embedding_vector, 
#         #                             payload=metadata
#         #                         ) ],
#         #                 )
#         #     operation_counts["ADD"] += 1
#         #     returned_memories.append(
#         #         {
#         #             "id": memory_id,
#         #             "memory": msg_content,
#         #             "event": "ADD",
#         #             # "actor_id": actor_name if actor_name else None,
#         #             "role": message_dict["role"],
#         #         }
#         #     )

#         parsed_messages = parse_messages(session)
#         user_prompt = f"Input:\n{parsed_messages}"
#         llm_response = openai_client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format={"type": "json_object"},
#         )
#         response = llm_response.choices[0].message.content
#         # print(f"LLM 返回的原始响应: {response}")
#         if response == '{"facts" : []}':
#             # print("parsed_messages:", parsed_messages)
#             continue
#         try:
#             response = remove_code_blocks(response)
#             if not response.strip():
#                 new_retrieved_facts = []
#             else:
#                 try:
#                     new_retrieved_facts = json.loads(response)["facts"]
#                 except json.JSONDecodeError:
#                     extracted_json = extract_json(response)
#                     new_retrieved_facts = json.loads(extracted_json)["facts"]
#         except Exception as e:
#             print(f"Error in new_retrieved_facts: {e}")
#             new_retrieved_facts = []
#         # print(f"新检索到的事实: {new_retrieved_facts}")
#         for fact in new_retrieved_facts:
#             embedding_vector = get_embedding(openai_client, fact, dimension=dimension)
#             memory_id = str(uuid.uuid4())
#             vect_store_client.upsert(
#                 collection_name=collection_name, wait=True,
#                 points=[ PointStruct(
#                             id=memory_id, vector=embedding_vector, 
#                             payload={ "data": fact, "created_at": date_string.isoformat()}
#                         ) ],
#                 )
#             operation_counts["ADD"] += 1
#             returned_memories.append(
#                 {
#                     "id": memory_id,
#                     "memory": fact,
#                     "event": "ADD",
#                 }
#             )
#         print(f"最终返回的记忆操作结果: {returned_memories}")
        
#     return operation_counts


# def response_user(line):
#     question = line.get("question")
#     question_date = line.get("question_date")
#     question_date = question_date + " UTC"
#     question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
#     question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
#     question = line.get("question")
#     question_vector = get_embedding(openai_client, question, dimension=dimension)
#     retrieved_memories = search(collection_name, vect_store_client, question_vector, top_k=topk)
#     # context = "\n".join([mem.payload.get("data", "") for mem in retrieved_memories])
#     memories_str = (
#             "\n".join(
#                 f"- {mem.payload.get('created_at', '')}: {mem.payload.get('data', '')}"
#                 for mem in retrieved_memories
#             )
#         )
#     response = generate_response(openai_client, question, question_date_string, memories_str)
#     answer = response.choices[0].message.content

#     return answer

# def process_and_evaluate_user(line, user_index, client, infer):
#     """
#     封装单个用户的所有处理步骤，以便并行执行。
#     返回一个包含所有统计信息的字典。
#     """
#     try:
#         if infer:
#             memory_counts = process_user_memory_infer(line)
#         else:
#             memory_counts = process_user_memory(line)
        
#         answer = response_user(line)
#         golden_answer = line.get("answer") 
#         question = line.get("question")
        
#         is_correct = lme_grader(client, question, golden_answer, answer)
        
#         return {
#             "index": user_index,
#             "is_correct": is_correct,
#             "counts": memory_counts,
#             "question": question,
#             "answer": answer,
#             "golden_answer": golden_answer
#         }
#     except Exception as e:
#         print(f"Error processing user {user_index} ({line.get('question', 'Unknown')[:20]}...): {e}")
#         return {
#             "index": user_index,
#             "is_correct": False,
#             "counts": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0},
#             "question": line.get("question", "N/A")
#         }


# if __name__ == "__main__":
#     # 清空并重新创建 Qdrant 集合
#     try:
#         if vect_store_client.collection_exists(collection_name=collection_name):
#             vect_store_client.delete_collection(collection_name=collection_name)
#         vect_store_client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=dimension, distance=Distance.DOT),
#         )
#     except Exception as e:
#         print(f"清空 Qdrant 集合失败: {e}. 请检查 Qdrant 客户端连接。")
#         exit()

#     with open("./data/longmemeval_s_cleaned.json", "r") as f:
#         lines = json.load(f)[:50]
    
#     print(f"已加载 {len(lines)} 个用户/问题。")

#     user_detail_results = [] 
#     total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    
#     MAX_WORKERS = 10
#     infer = False

#     futures = []

#     print(f"开始使用 {MAX_WORKERS} 个线程并行处理...")
    
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         for idx, line in enumerate(lines):
#             future = executor.submit(process_and_evaluate_user, line, idx + 1, openai_client, infer=infer)
#             futures.append(future)
        
#         for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
#             result = future.result()
#             user_detail_results.append(result)

#     user_detail_results.sort(key=lambda x: x.get("index", 0))

#     correct_count = 0
#     total_evaluated = len(user_detail_results)

#     for res in user_detail_results:
#         if res.get("is_correct"):
#             correct_count += 1
        
#         counts = res.get("counts", {})
#         for key in total_memory_counts:
#             total_memory_counts[key] += counts.get(key, 0)

#     print("\n\n==================================================")
#     print("             🎯 最终评估结果") 
#     print("==================================================")

#     if total_evaluated > 0:
#         final_accuracy = correct_count / total_evaluated
#         print(f"总评估问题数: {total_evaluated}")
#         print(f"正确回答数: {correct_count}")
#         print(f"最终总准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
#     else:
#         print("没有评估任何问题。")
#     print("==================================================")

#     print("\n\n==================================================")
#     print("        📊 详细记忆操作统计 (按用户)")
#     print("==================================================")

#     for res in user_detail_results:
#         user_index = res["index"]
#         is_correct = res["is_correct"]
#         counts = res["counts"]
#         question = res["question"]
#         answer = res.get("answer", "N/A")
#         golden_answer = res.get("golden_answer", "N/A")
#         status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
#         print(f"\n--- 用户/问题 {user_index} ---")
#         print(f"  问题: {question[:60]}...")
#         print(f"  模型回答: {answer[:60]}...")
#         print(f"  标准答案: {golden_answer[:60]}...")
#         print(f"  评估结果: {status}")
#         print(f"  记忆操作: ADD={counts.get('ADD', 0)}, UPDATE={counts.get('UPDATE', 0)}, DELETE={counts.get('DELETE', 0)}, NONE={counts.get('NONE', 0)}")

#     print("\n--- 所有用户的记忆操作总览 ---")
#     print(f"  ADD (新增):    {total_memory_counts['ADD']}")
#     print(f"  UPDATE (更新): {total_memory_counts['UPDATE']}")
#     print(f"  DELETE (删除): {total_memory_counts['DELETE']}")
#     print(f"  NONE (无操作): {total_memory_counts['NONE']}")
#     print("==================================================")


import os
import shutil
import random
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
# --- 1. 配置 ---
DB_PATH = "./qdrant_db" # 临时的数据库路径
COLLECTION_NAME = "test_update_collection"
DIMENSION = 4  # 使用一个小的维度来快速测试
ITEM_ID = str(uuid.uuid4()) # 我们将要更新的固定 ID

# --- 2. 初始化 Qdrant 客户端和集合 ---
# 确保我们从一个干净的状态开始
# if os.path.exists(DB_PATH):
#     print(f"发现旧的测试数据库，正在删除: {DB_PATH}")
#     shutil.rmtree(DB_PATH)

# 使用 Qdrant 文件存储
client = QdrantClient(path=DB_PATH)

print(f"\n创建 Qdrant 集合: {COLLECTION_NAME}")
# 使用 recreate_collection 来确保每次运行都是全新的
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    # 维度和距离单位应与您的主脚本匹配
    vectors_config=VectorParams(size=DIMENSION, distance=Distance.DOT) 
)
print("集合创建成功。")

try:
    # --- 步骤 1: 插入 (ADD) - 版本 1 ---
    print("\n--- 步骤 1: 插入 '版本 1' 数据 ---")
    v1_text = "这是原始版本的数据"
    v1_vector = [random.random() for _ in range(DIMENSION)]
    v1_created_at = datetime.now(timezone.utc).isoformat()
    
    v1_payload = {
        "data": v1_text,
        "created_at": v1_created_at
        # 注意：v1 没有 updated_at 字段
    }
    
    v1_point = PointStruct(id=ITEM_ID, vector=v1_vector, payload=v1_payload)
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[v1_point],
        wait=True # 等待操作完成
    )
    print(f"  > 已插入 ID: {ITEM_ID}")
    print(f"  > Payload: {v1_payload}")

    # --- 步骤 2: 验证 (VERIFY) - 版本 1 ---
    print("\n--- 步骤 2: 验证 '版本 1' 数据 ---")
    retrieved_v1_list = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[ITEM_ID],
        with_payload=True
    )
    
    assert len(retrieved_v1_list) == 1, "错误：插入后未能检索到数据！"
    retrieved_v1_point = retrieved_v1_list[0]
    
    print(f"  > 检索结果: {retrieved_v1_point}")
    assert retrieved_v1_point.payload["data"] == v1_text, "错误：v1 的 data 字段不匹配！"
    assert retrieved_v1_point.payload["created_at"] == v1_created_at, "错误：v1 的 created_at 字段不匹配！"
    
    print("  > 验证成功： '版本 1' 数据已正确存储。")

    # --- 步骤 3: 更新 (UPDATE) - 版本 2 ---
    print("\n--- 步骤 3: 更新 (Upsert) '版本 2' 数据 (模拟您的脚本) ---")
    
    # 1. 获取旧的 payload
    original_payload = retrieved_v1_point.payload
    
    # 2. 准备新数据
    v2_text = "这是已更新的版本"
    v2_vector = [random.random() for _ in range(DIMENSION)] # 模拟新的嵌入向量
    v2_updated_at = datetime.now(timezone.utc).isoformat()

    # 3. 创建新的 payload，修改 data 和 updated_at，但保留 created_at
    v2_payload = original_payload.copy() # 复制旧 payload
    v2_payload["data"] = v2_text         # 更新 data
    v2_payload["updated_at"] = v2_updated_at # 添加 updated_at
    
    # 4. 用新向量和新 payload 覆盖旧的 ID
    v2_point = PointStruct(id=ITEM_ID, vector=v2_vector, payload=v2_payload)
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[v2_point],
        wait=True
    )
    print(f"  > 已更新 ID: {ITEM_ID}")
    print(f"  > 新 Payload: {v2_payload}")

    # --- 步骤 4: 验证 (VERIFY) - 版本 2 ---
    print("\n--- 步骤 4: 验证 '版本 2' 数据 ---")
    retrieved_v2_list = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[ITEM_ID],
        with_payload=True
    )
    
    assert len(retrieved_v2_list) == 1, "错误：更新后未能检索到数据！"
    retrieved_v2_point = retrieved_v2_list[0]

    print(f"  > 检索结果: {retrieved_v2_point}")
    
    # 验证数据是否已更新
    assert retrieved_v2_point.payload["data"] == v2_text, "错误：v2 的 data 字段未更新！"
    # 验证 created_at 是否被保留
    assert retrieved_v2_point.payload["created_at"] == v1_created_at, "错误：created_at 字段在更新时丢失了！"
    # 验证 updated_at 是否已添加
    assert retrieved_v2_point.payload["updated_at"] == v2_updated_at, "错误：updated_at 字段未正确添加！"
    
    print("\n==============================================")
    print("✅ 验证成功！Qdrant 的 '读取-修改-写回' 更新逻辑工作正常。")
    print("==============================================")

finally:
    # --- 步骤 5: 清理 ---
    print(f"\n--- 步骤 5: 清理 ---")
    # client.close()
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree(DB_PATH)
    #     print(f"已删除临时 Qdrant 数据库: {DB_PATH}")