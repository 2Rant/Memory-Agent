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
# vect_store_client = QdrantClient(url=os.getenv("QDRANT_URL"),
#                                   api_key=os.getenv("QDRANT_API_KEY"))
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
#         print(f"用于记忆更新的旧事实: {retrieved_old_facts}") 

#         if new_retrieved_facts:
#             memory_action_prompt = get_update_memory_messages(retrieved_old_facts, new_retrieved_facts)
#             # print("Memory Action Prompt:", memory_action_prompt)
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
#                         collection_name=collection_name, 
#                         wait=True,
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
#                             wait=True,
#                             points=[ PointStruct(
#                                         id=temp_uuid_mapping.get(resp.get("id")), 
#                                         vector=embedding_vector, 
#                                         payload=result.payload
#                                     ) ],
#                         )
#                         returned_memories.append( 
#                             {
#                                 # "id": temp_uuid_mapping.get(resp.get("id"), "update_get_id_error"),
#                                 "id": resp.get("id"),
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
#                             wait=True,
#                             points_selector=PointIdsList(
#                                 points=[temp_uuid_mapping.get(resp.get("id"))]
#                             ),
#                         )
#                         returned_memories.append(
#                             {
#                                 # "id": temp_uuid_mapping.get(resp.get("id"), "delete_get_id_error"),
#                                 "id": resp.get("id"),
#                                 "memory": action_text,
#                                 "event": event_type,
#                             }
#                         )
#                     elif event_type == "NONE":
#                         returned_memories.append(
#                             {
#                                 # "id": temp_uuid_mapping.get(resp.get("id"), "none_get_id_error"),
#                                 "id": resp.get("id"),
#                                 "memory": action_text,
#                                 "event": event_type,
#                             }
#                         )
#                 except Exception as e:
#                     print("==================================================")
#                     print(f"Error processing memory action {resp}: {e}")   
#                     print(f"完整响应内容: {new_memories_with_actions}")
#                     print(f"临时 UUID 映射: {temp_uuid_mapping}") 
#                     print(f"检索到的旧事实: {retrieved_old_facts}")
#                     print("==================================================")

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
#                 collection_name=collection_name, 
#                 wait=True,
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
#         # print(f"最终返回的记忆操作结果: {returned_memories}")
        
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
#     infer = True

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
#         print(f"  模型回答: {answer}...")
#         print(f"  标准答案: {golden_answer}...")
#         print(f"  评估结果: {status}")
#         print(f"  记忆操作: ADD={counts.get('ADD', 0)}, UPDATE={counts.get('UPDATE', 0)}, DELETE={counts.get('DELETE', 0)}, NONE={counts.get('NONE', 0)}")

#     print("\n--- 所有用户的记忆操作总览 ---")
#     print(f"  ADD (新增):    {total_memory_counts['ADD']}")
#     print(f"  UPDATE (更新): {total_memory_counts['UPDATE']}")
#     print(f"  DELETE (删除): {total_memory_counts['DELETE']}")
#     print(f"  NONE (无操作): {total_memory_counts['NONE']}")
#     print("==================================================")




# import json
# import os
# from datetime import datetime
# from typing import List, Dict, Any
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# # 模拟配置，请替换为你的真实 Key
# # os.environ["OPENAI_API_KEY"] = "sk-..." 
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
#                 base_url=os.getenv("OPENAI_BASE_URL"))

# # ==============================================================================
# # 1. Prompt 定义
# # ==============================================================================

# # 注意：我对你的原始 Prompt 做了微调，增加了 {core_memory_block} 插槽
# # 这样 LLM 才能看到"旧记忆"，从而实现指代消解和去重。
# MEM_READER_PROMPT_TEMPLATE = """You are a Personal Information Organizer. Your goal is to extract RELEVANT facts from the user conversation.

# ### EXISTING MEMORY CONTEXT ###
# The following information is ALREADY known. Use this to:
# 1. Resolve pronouns (e.g., "He" -> "Tom").
# 2. Ignore facts that are already exactly stored.
# 3. Provide context for new preferences.

# [Start of Memory]
# {core_memory_block}
# [End of Memory]

# ### INSTRUCTIONS ###
# Extract new facts based on the user input.
# - If the input is just a greeting or irrelevant, return empty facts.
# - Detect the language of the user input and record facts in the same language.
# - Return format must be JSON: {{"facts": ["fact string 1", "fact string 2"]}}

# ### Types of Information to Remember ###
# 1. Personal Preferences (Food, hobbies, etc.)
# 2. Personal Details (Names, relationships)
# 3. Plans and Intentions
# 4. Professional Details
# 5. Health/Wellness

# ### CONVERSATION ###
# User: {user_input}
# """

# # Agent 负责根据新提取的事实和旧记忆，决定具体操作 (ADD, UPDATE, DELETE, NONE)
# AGENT_INFER_PROMPT_TEMPLATE = """You are the Memory Manager Agent.
# You have two inputs:
# 1. Existing Memories (The current state of the database)
# 2. New Extracted Facts (What the user just said, processed by MemReader)

# Your task is to determine the correct OPERATION for each new fact.
# Available Operations:
# - ADD: Information is completely new.
# - UPDATE: Information conflicts with or refines an existing memory (provide the old_memory_id).
# - DELETE: User explicitly asks to forget something.
# - NONE: Information is redundant or already exists.

# ### Existing Memories ###
# {memory_list_json}

# ### New Extracted Facts ###
# {new_facts_json}

# ### OUTPUT FORMAT ###
# Return a JSON object with a key "operations". Example:
# {{
#     "operations": [
#         {{"action": "ADD", "content": "Likes sushi"}},
#         {{"action": "UPDATE", "target_memory_id": "mem_01", "new_content": "Lives in New York"}},
#         {{"action": "NONE", "reason": "Already known"}}
#     ]
# }}
# """

# # ==============================================================================
# # 2. Pipeline 类实现
# # ==============================================================================

# class MemoryPipeline:
#     def __init__(self):
#         self.model = "gpt-4o-mini"  # 建议使用 gpt-4o 或 gpt-3.5-turbo

#     def _get_completion(self, prompt: str) -> str:
#         """简单的 OpenAI 调用封装"""
#         response = client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"}, # 强制 JSON 输出
#             temperature=0
#         )
#         return response.choices[0].message.content

#     def mem_reader(self, user_input: str, core_memory: List[Dict]) -> List[str]:
#         """
#         阶段 1: MemReader
#         功能: 结合 Core Memory 上下文，从对话中提取纯净的 Facts
#         """
#         # 1. 格式化旧记忆给 MemReader 看
#         if not core_memory:
#             memory_str = "No existing memory."
#         else:
#             # 简化展示，只给内容
#             memory_str = "\n".join([f"- {m['content']}" for m in core_memory])

#         # 2. 填充 Prompt
#         formatted_prompt = MEM_READER_PROMPT_TEMPLATE.format(
#             core_memory_block=memory_str,
#             user_input=user_input,
#             datetime=datetime.now().strftime("%Y-%m-%d")
#         )

#         # 3. 调用 LLM
#         print(f"   [MemReader] Processing input: '{user_input}'...")
#         response_json = self._get_completion(formatted_prompt)
        
#         try:
#             data = json.loads(response_json)
#             facts = data.get("facts", [])
#             print(f"   [MemReader] Extracted: {facts}")
#             return facts
#         except Exception as e:
#             print(f"Error parsing MemReader output: {e}")
#             return []

#     def agent_infer(self, new_facts: List[str], core_memory: List[Dict]) -> Dict:
#         """
#         阶段 2: Agent Inference
#         功能: 对比 New Facts 和 Core Memory，生成操作指令 (Add/Update/Delete)
#         """
#         if not new_facts:
#             return {"operations": []}

#         # 1. 准备数据
#         # 给 Agent 完整的带有 ID 的记忆，以便它能指定 Update 哪一条
#         memory_json = json.dumps(core_memory, indent=2, ensure_ascii=False)
#         facts_json = json.dumps(new_facts, indent=2, ensure_ascii=False)

#         # 2. 填充 Prompt
#         formatted_prompt = AGENT_INFER_PROMPT_TEMPLATE.format(
#             memory_list_json=memory_json,
#             new_facts_json=facts_json
#         )

#         # 3. 调用 LLM
#         print(f"   [Agent] Inferring operations...")
#         response_json = self._get_completion(formatted_prompt)
        
#         try:
#             return json.loads(response_json)
#         except Exception as e:
#             print(f"Error parsing Agent output: {e}")
#             return {"operations": []}

#     def run(self, user_input: str, core_memory: List[Dict]):
#         """执行完整流程"""
#         print(f"\n--- Pipeline Start ---")
        
#         # Step 1: MemReader (Conversation + Memory -> Facts)
#         extracted_facts = self.mem_reader(user_input, core_memory)
        
#         # Step 2: Agent Inference (Facts + Memory -> Operations)
#         results = self.agent_infer(extracted_facts, core_memory)
        
#         print(f"   [Result] Operations: {json.dumps(results, indent=2, ensure_ascii=False)}")
#         print(f"--- Pipeline End ---\n")
#         return results

# # ==============================================================================
# # 3. 验证实验 (A/B Test)
# # ==============================================================================

# if __name__ == "__main__":
#     pipeline = MemoryPipeline()

#     # 模拟当前的 Core Memory (带有 ID)
#     current_memory = [
#         {"id": "mem_01", "content": "User's son is named Tom."},
#         {"id": "mem_02", "content": "User lives in Shanghai."},
#         {"id": "mem_03", "content": "User works as a Python Developer."}
#     ]

#     print("=== 实验 1: 验证 Core Memory 对 MemReader 的影响 (指代消解) ===")
#     user_input = "It is his birthday tomorrow."
    
#     print("\n>>> Group A: 假如没有 Core Memory (空列表)")
#     pipeline.run(user_input, core_memory=[]) 
#     # 预期 MemReader 输出: "It is his birthday tomorrow" (模糊，Agent 无法处理)

#     print("\n>>> Group B: 注入 Core Memory")
#     pipeline.run(user_input, core_memory=current_memory)
#     # 预期 MemReader 输出: "It is Tom's birthday tomorrow" (清晰，Agent 生成 ADD 操作)

#     print("\n" + "="*50 + "\n")

#     print("=== 实验 2: 验证 Update/Conflict 逻辑 ===")
#     user_input_2 = "I moved to Beijing yesterday."
    
#     # 这里的 MemReader 应该提取出 "User moved to Beijing"
#     # Agent 应该检测到与 mem_02 (Shanghai) 冲突，生成 UPDATE 操作
#     pipeline.run(user_input_2, core_memory=current_memory)


