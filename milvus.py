from copy import deepcopy
import uuid
# Qdrant imports REMOVED
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from qdrant_client.models import PointStruct

# Milvus imports ADDED
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

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
topk = 5
system_prompt = FACT_RETRIEVAL_PROMPT

# --- Milvus Connection and Collection Setup ---
# Connect to Milvus Lite (file-based)
connections.connect(alias="default", uri="./milvus.db")

if not utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' not found. Creating...")
    # Define the schema
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36)
    vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    data_field = FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=65535) # Max length for text
    created_at_field = FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=100)
    updated_at_field = FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=100, default_value="")

    schema = CollectionSchema(
        fields=[id_field, vector_field, data_field, created_at_field, updated_at_field],
        description="LME Test Collection",
        enable_dynamic_field=False
    )
    
    # Create collection
    vect_store_client = Collection(name=collection_name, schema=schema)

    # Create index
    index_params = {
        "metric_type": "IP",  # Qdrant 'DOT' is equivalent to Milvus 'IP' (Inner Product)
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    vect_store_client.create_index(field_name="vector", index_params=index_params)
    print("Collection created and index built.")
else:
    print(f"Collection '{collection_name}' already exists.")
    vect_store_client = Collection(name=collection_name)

# Load the collection into memory for searching
vect_store_client.load()
# -----------------------------------------------


def search(collection_name, vect_store_client, query_vector, top_k=5):
    # 'vect_store_client' is now the Milvus Collection object
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }
    
    results = vect_store_client.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["data", "created_at", "updated_at"] # Specify fields to return
    )
    
    # --- Shim: Format Milvus results to look like Qdrant's PointStruct ---
    # This is so the rest of the code (mem.payload.get) doesn't break
    # 这是一个“模拟类”，用来伪装 Milvus 的返回结果，使其看起来像 Qdrant 的结果
    class MockPoint:
        def __init__(self, id, payload, score):
            self.id = id
            self.payload = payload
            self.score = score

    formatted_results = []
    if results and results[0]:
        for hit in results[0]:
            payload = {
                "data": hit.entity.get("data"),
                "created_at": hit.entity.get("created_at"),
                "updated_at": hit.entity.get("updated_at")
            }
            formatted_results.append(MockPoint(id=hit.id, payload=payload, score=hit.distance))
    # --------------------------------------------------------------------
    
    return formatted_results

# The 'insert' function was defined but not used in the original script.
# I'm removing it to avoid confusion. The script uses 'upsert' directly.
# def insert(collection_name, vect_store_client, vectors, payloads=None):
#     ...

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
                
                # Qdrant upsert (commented out for reference)
                # operation_info = vect_store_client.upsert(
                # ...
                # )
                
                # Search using the new Milvus-compatible 'search' function
                existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
                
                for mem in existing_memories:
                    retrieved_old_facts.append({"id": mem.id, "text": mem.payload.get("data", "")})
                    print("mem:", mem) # mem is now a MockPoint object
        except Exception as e:
            print(f"生成嵌入或搜索时出错：{e}")


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
                        
                        # --- Milvus ADD (Upsert) ---
                        data_to_insert = [{
                            "id": memory_id,
                            "vector": embedding_vector,
                            "data": action_text,
                            "created_at": date_string.isoformat(),
                            "updated_at": "" # Empty default
                        }]
                        operation_info = vect_store_client.upsert(data=data_to_insert)
                        vect_store_client.flush() # Ensure data is written
                        # ---------------------------
                        
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":
                        
                        # --- Milvus Retrieve (Query) ---
                        # Shim class to mimic Qdrant's retrieve output
                        # 另一个模拟类，用于伪装 query 的返回结果
                        class MockRetrievePoint:
                            def __init__(self, data_dict, record_id):
                                self.id = record_id
                                self.payload = {
                                    "data": data_dict.get("data"),
                                    "created_at": data_dict.get("created_at"),
                                    "updated_at": data_dict.get("updated_at")
                                }
                        
                        record_id = temp_uuid_mapping.get(resp.get("id"))
                        points_data = vect_store_client.query(
                            expr=f"id == '{record_id}'",
                            output_fields=["data", "created_at", "updated_at"]
                        )
                        result = MockRetrievePoint(points_data[0], record_id) if points_data else None
                        # ---------------------------------

                        if result:
                            old_memory = result.payload.get("data", "")
                        else:
                            old_memory = ""
                        created_at = result.payload.get("created_at", "") if result else ""
                        new_updated_at = date_string.isoformat()
                        
                        # --- Milvus UPDATE (Upsert) ---
                        data_to_update = [{
                            "id": temp_uuid_mapping.get(resp.get("id")),
                            "vector": embedding_vector,
                            "data": action_text,
                            "created_at": created_at, # Keep original creation date
                            "updated_at": new_updated_at
                        }]
                        vect_store_client.upsert(data=data_to_update)
                        vect_store_client.flush() # Ensure data is written
                        # ----------------------------

                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping.get(resp.get("id")),
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": old_memory,
                            }
                        )
                    elif event_type == "DELETE":
                        
                        # --- Milvus DELETE ---
                        record_id = temp_uuid_mapping.get(resp.get("id"))
                        vect_store_client.delete(expr=f"id == '{record_id}'")
                        vect_store_client.flush() # Ensure data is deleted
                        # ---------------------
                        
                        returned_memories.append(
                            {
                                "id": resp.get("id"),
                                "memory": action_text,
                                "event": event_type,
                            }
                        )

                    elif event_type == "NONE":
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
    
    # This call now uses the Milvus-backed 'search' function
    # 这里的调用现在使用由 Milvus 支持的 'search' 函数
    retrieved_memories = search(collection_name, vect_store_client, question_vector, top_k=topk)
    
    # This part works without change because our 'search' function
    # returns objects that mimic Qdrant's (mem.payload.get(...))
    # 这部分代码无需更改，因为我们的 'search' 函数返回了模拟 Qdrant 格式的对象
    memories_str = (
            "\n".join(
                f"- {mem.payload.get('created_at', '')}: {mem.payload.get('data', '')}"
                for mem in retrieved_memories
            )
        )
    response = generate_response(openai_client, question, question_date_string, memories_str)
    answer = response.choices[0].message.content

    return answer

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
print("                    🎯 最终评估结果")
print("==================================================")

if total_evaluated > 0:
    final_accuracy = correct_count / total_evaluated
    print(f"总评估问题数: {total_evaluated}")
    print(f"正确回答数: {correct_count}")
    print(f"最终总准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
else:
    print("没有评估任何问题。")
print("==================================================")

# Finally, disconnect from Milvus
# 最后，断开与 Milvus 的连接
connections.disconnect(alias="default")
print("\nDisconnected from Milvus.")