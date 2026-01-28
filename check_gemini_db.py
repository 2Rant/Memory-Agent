from pymilvus import connections, Collection, utility
import os
from dotenv import load_dotenv

load_dotenv()

def check_db():
    milvus_uri = os.getenv("MILVUS_URI")
    user = os.getenv("MILVUS_USER_NAME")
    password = os.getenv("MILVUS_PASSWORD")
    
    connections.connect(uri=milvus_uri, user=user, password=password)
    
    # 检查 memories_longmemeval_gemini
    col_name = "memories_longmemeval_gemini"
    if col_name not in utility.list_collections():
        print(f"Collection {col_name} not found")
        return
        
    col = Collection(col_name)
    col.load()
    
    # 取 user_0
    res = col.query(expr="user_id == 'user_0'", output_fields=["user_id", "content"], limit=5)
    print(f"Samples for {col_name} (user_0):")
    for r in res:
        print(f"Content: {r['content'][:100]}...")

if __name__ == "__main__":
    check_db()
