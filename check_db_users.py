from pymilvus import connections, Collection, utility
import os
from dotenv import load_dotenv

load_dotenv()

def check_db():
    milvus_uri = os.getenv("MILVUS_URI")
    user = os.getenv("MILVUS_USER_NAME")
    password = os.getenv("MILVUS_PASSWORD")
    
    connections.connect(uri=milvus_uri, user=user, password=password)
    
    # 获取所有集合名称
    all_collections = utility.list_collections()
    print(f"Available Collections: {all_collections}")
    
    col_name = "memories_longmemeval_v1"
    if col_name not in all_collections:
        print(f"Collection {col_name} not found")
        return
        
    col = Collection(col_name)
    col.load()
    
    # 随机取几条数据看看 user_id
    res = col.query(expr="memory_id != ''", output_fields=["user_id", "content"], limit=10)
    print("\nDatabase Samples:")
    for r in res:
        print(f"User: {r['user_id']} | Content: {r['content'][:50]}...")

if __name__ == "__main__":
    check_db()
