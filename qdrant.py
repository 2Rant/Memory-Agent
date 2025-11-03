import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from utils import get_embedding, parse_messages, FACT_RETRIEVAL_PROMPT, remove_code_blocks, extract_json, get_update_memory_messages
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import pytz
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
dimension=4
collection_name = "test_collection1"
vect_store_client = QdrantClient(path="./qdrant_db")

if not os.path.exists("./qdrant_db/collection/" + collection_name):
    vect_store_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.DOT),
    )

with open("/mnt/afs/codes/ljl/MemOS-pipeline/evaluation/data/prefeval/pref_processed.jsonl", "r") as f:
    lines = f.readlines()[:1]
# lines = [
#     {
#         # 这就是您的后续代码 (line["messages"]) 需要访问的键
#         "messages": [{"role": "user", "content": "Hello, this is a sample text for embedding generation."}],
#         "id": 123,
#         "other_info": "mock_data"
#     }
# ]

for idx, line in enumerate(lines):
    line = json.loads(line)
    parsed_messages = parse_messages(line["conversation"])
    system_prompt = FACT_RETRIEVAL_PROMPT
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

def search(collection_name, vect_store_client, query_vector, top_k=5):
    search_result = vect_store_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        with_payload=True,
        limit=top_k
    ).points
    # print(search_result)
    return search_result

retrieved_old_facts = []
try:
    for fact in new_retrieved_facts:
        embedding_vector = get_embedding(openai_client, fact, dimension=dimension)

        # print(f"原始文本: {fact}")
        # print(f"生成的嵌入维度: {len(embedding_vector)}")
        # print(f"嵌入向量的前 5 个数值: {embedding_vector[:5]}")

        operation_info = vect_store_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(id=str(uuid.uuid4()), 
                        vector=embedding_vector, 
                        payload={
                                "data": fact, 
                                "created_at": datetime.now(pytz.timezone('Asia/Shanghai')).isoformat()
                                }
                        )
                        ],
        )
        # print(operation_info)
        existing_memories = search(collection_name, vect_store_client, embedding_vector, top_k=5)
        # print(f"检索到的记忆点: {existing_memories}")
        for mem in existing_memories:
            retrieved_old_facts.append(mem.payload["data"])
            print("mem:", mem)
    # print(f"检索到的旧事实: {retrieved_old_facts}")
except Exception as e:
    print(f"生成嵌入时出错，请检查您的 API Key 和网络连接：{e}")


# # 验证点是否成功添加和存储
# points_data = client.retrieve(
#     collection_name=collection_name,
#     ids=[1, 2, 3, 4, 5, 6], # 您插入的点的ID
#     with_payload=True, # 确保返回 payload (城市信息)

# )

# print("\n--- 检索到的数据点 ---")
# for point in points_data:
#     print(f"ID: {point.id}, Payload: {point.payload}")
# print("-----------------------")
def insert(collection_name, vect_store_client, vectors, payloads=None):
    points = [
        PointStruct(id=idx, vector=vector, payload=payloads[idx])
        for idx, vector in enumerate(vectors)
    ]
    vect_store_client.upsert(
        collection_name=collection_name,
        points=points
    )


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
            new_memories_with_action = {}
        else:
            response = remove_code_blocks(update_response)
            