
from copy import deepcopy
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from utils import get_embedding, parse_messages, FACT_RETRIEVAL_PROMPT, remove_code_blocks, extract_json, get_update_memory_messages, LME_JUDGE_MODEL_TEMPLATE, LME_ANSWER_PROMPT 
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
    lines = json.load(f)[:2]


# for idx, line in enumerate(lines):
    # line = json.loads(line)
    # parsed_messages = parse_messages(line["conversation"])
def process_user_memory(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    question = line.get("question")
    golden_answer = line.get("golden_answer")

    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        parsed_messages = parse_messages(session) 
        print("parsed_messages:", parsed_messages)
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
        print(f"LLM 返回的原始响应: {response}")
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
        print(f"新检索到的事实: {new_retrieved_facts}")

        retrieved_old_facts = []
        new_message_embeddings = {}
        try:
            for fact in new_retrieved_facts:
                embedding_vector = get_embedding(openai_client, fact, dimension=dimension)
                new_message_embeddings[fact] = embedding_vector
                # print(f"原始文本: {fact}")
                # print(f"生成的嵌入维度: {len(embedding_vector)}")
                # print(f"嵌入向量的前 5 个数值: {embedding_vector[:5]}")
            
                # operation_info = vect_store_client.upsert(
                # collection_name=collection_name,
                # wait=True,
                # points=[
                #     PointStruct(id=str(uuid.uuid4()), 
                #                 vector=embedding_vector, 
                #                 payload={
                #                         "data": fact, 
                #                         "created_at": datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
                #                         }
                #                 )
                #                 ],
                # )
                # print(operation_info)

                existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
                # print(f"检索到的记忆点: {existing_memories}")
                for mem in existing_memories:
                    retrieved_old_facts.append({"id": mem.id, "text": mem.payload.get("data", "")})
                    print("mem:", mem)
            # print(f"检索到的旧事实: {retrieved_old_facts}")
        except Exception as e:
            print(f"生成嵌入时出错，请检查您的 API Key 和网络连接：{e}")


# # 验证点是否成功添加和存储
# points_data = vect_store_client.retrieve(
#     collection_name=collection_name,
#     ids=[pp], # 您插入的点的ID
#     with_payload=True, # 确保返回 payload (城市信息)

# )

# print("\n--- 检索到的数据点 ---")
# for point in points_data:
#     print("point:", point)
#     print(f"ID: {point.id}, Payload: {point.payload}")
# print("-----------------------")

        unique_data = {}
        for item in retrieved_old_facts:
            unique_data[item["id"]] = item
        retrieved_old_facts = list(unique_data.values())

        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_facts):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_facts[idx]["id"] = str(idx)
        print(f"临时 UUID 映射: {temp_uuid_mapping}")
        print(f"用于记忆更新的旧事实: {retrieved_old_facts}")
        if new_retrieved_facts:
            memory_action_prompt = get_update_memory_messages(retrieved_old_facts, new_retrieved_facts)
            # print("用于更新记忆的提示:", memory_action_prompt)
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": memory_action_prompt}],
                response_format={"type": "json_object"},
            )
            update_response = response.choices[0].message.content

            try:
                if not update_response.strip() or not update_response:
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
        try:
            for resp in new_memories_with_actions.get("memory", []):
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        print("Skipping memory entry because of empty `text` field.")
                        continue
                    embedding_vector = get_embedding(openai_client, action_text, dimension=dimension)
                    event_type = resp.get("event")
                    if event_type == "ADD":
                        memory_id = str(uuid.uuid4())
                        # memory_id = self._create_memory(
                        #     data=action_text,
                        #     existing_embeddings=new_message_embeddings,
                        #     metadata=deepcopy(metadata),
                        # )
                        operation_info = vect_store_client.upsert(
                        collection_name=collection_name,
                        wait=True,
                        points=[
                                PointStruct(
                                        id=memory_id, 
                                        vector=embedding_vector, 
                                        payload={
                                            "data": action_text, 
                                            # "created_at": datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
                                            "created_at": date_string.isoformat()
                                        })
                                ],
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
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
                        created_at = result.payload.get("created_at", "") if result else ""
                        # new_updated_at = datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
                        new_updated_at = date_string.isoformat()
                        result.payload["data"] = action_text
                        result.payload["updated_at"] = new_updated_at
                        vect_store_client.upsert(
                            collection_name=collection_name,
                            points=[
                                PointStruct(
                                    id=temp_uuid_mapping.get(resp.get("id")), 
                                    vector=embedding_vector, 
                                    payload=result.payload
                                )
                            ],
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping.get(resp.get("id")),
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": old_memory,
                            }
                        )
                    elif event_type == "DELETE":
                        vect_store_client.delete(
                            collection_name=collection_name,
                            points_selector={
                                "points": [temp_uuid_mapping.get(resp.get("id"))],
                            }
                        )
                        returned_memories.append(
                            {
                                "id": resp.get("id"),
                                "memory": action_text,
                                "event": event_type,
                            }
                        )

                    elif event_type == "NONE":
                        # memory_id = temp_uuid_mapping.get(resp.get("id"))
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping.get(resp.get("id")),
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                except Exception as e:
                    print(f"Error processing memory action {resp}: {e}")       
        except Exception as e:
            print(f"Error iterating new_memories_with_actions: {e}")

        print(f"最终返回的记忆操作结果: {returned_memories}")

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

# for idx, line in enumerate(lines):
#     print(f"\n\n==== 处理第 {idx + 1} 个用户的记忆 ====")
#     process_user_memory(line)
#     print(f"\n\n==== 为第 {idx + 1} 个用户生成回答 ====")
#     answer = response_user(line)
#     print(f"生成的回答: {answer}")



# 1. 结果累加器
evaluation_results = []
correct_count = 0
total_evaluated = 0

with open("./data/longmemeval_s_cleaned.json", "r") as f:
    lines = json.load(f)[:2] # 仍只处理前 2 个用户

for idx, line in enumerate(lines):
    print(f"\n\n==== 处理第 {idx + 1} 个用户的记忆 (存储阶段) ====")
    process_user_memory(line)
    
    print(f"\n\n==== 为第 {idx + 1} 个用户生成回答 (检索阶段) ====")
    answer = response_user(line)
    golden_answer = line.get("golden_answer") # 获取黄金答案
    question = line.get("question") # 获取问题

    print(f"生成的回答: {answer}")
    print(f"黄金答案: {golden_answer}")
    
    # 2. 调用 Grader 进行评估
    is_correct = lme_grader(openai_client, question, golden_answer, answer)
    
    # 3. 统计结果
    total_evaluated += 1
    if is_correct:
        correct_count += 1
        evaluation_results.append(True)
    else:
        evaluation_results.append(False)

    print(f"LLM 评估结果: {'CORRECT' if is_correct else 'WRONG'}")
    print(f"当前累计准确率: {correct_count / total_evaluated:.4f} ({correct_count}/{total_evaluated})")


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