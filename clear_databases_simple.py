#!/usr/bin/env python3
"""
直接使用pymilvus库清空向量数据库集合

该脚本用于删除以下集合：
1. pipeline_v0.py创建的固定名称集合：memories_v0、facts_v0、chunks_v0
2. pipeline_v2.py创建的动态名称集合：
   - 测试模式：memories_test、facts_test、chunks_test
   - 评估模式：memories_eval、facts_eval、chunks_eval
   - 带数据集后缀的集合：memories_test_{dataset_name}等
"""

import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

# 从.env文件获取配置
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER_NAME = os.getenv("MILVUS_USER_NAME", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

print(f"📁 加载.env文件: {env_path}")
print(f"🔧 从.env读取的MILVUS_URI: {MILVUS_URI}")
print(f"🔧 从.env读取的MILVUS_USER_NAME: {MILVUS_USER_NAME}")

# 安装pymilvus库（如果未安装）
try:
    from pymilvus import MilvusClient
    print("✅ pymilvus库已安装")
except ImportError:
    print("⏳ 正在安装pymilvus库...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymilvus"])
    from pymilvus import MilvusClient
    print("✅ pymilvus库安装成功")


def delete_collection(client, collection_name):
    """删除指定集合，如果不存在则跳过"""
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"✅ 成功删除集合: {collection_name}")
        else:
            print(f"ℹ️  集合不存在: {collection_name}")
    except Exception as e:
        print(f"❌ 删除集合 {collection_name} 失败: {e}")


def main():
    """主函数"""
    print(f"\n📋 开始清空数据库...")
    print(f"   数据库URI: {MILVUS_URI}")
    
    # 连接到Milvus数据库
    try:
        client = MilvusClient(
            uri=MILVUS_URI,
            user=MILVUS_USER_NAME,
            password=MILVUS_PASSWORD
        )
        print("✅ 成功连接到Milvus数据库")
    except Exception as e:
        print(f"❌ 连接Milvus数据库失败: {e}")
        return 1
    
    # 要删除的集合名称
    collections_to_delete = [
        # # pipeline_v0.py创建的集合
        # "memories_v0",
        # "facts_v0",
        # "chunks_v0",
        
        # # pipeline_v2.py创建的集合（测试模式）
        # "memories_test",
        # "facts_test",
        # "chunks_test",
        
        # # pipeline_v2.py创建的集合（评估模式）
        # "memories_eval",
        # "facts_eval",
        # "chunks_eval",
        
        # # 带数据集后缀的集合（示例）
        # "memories_test_longmemeval",
        # "facts_test_longmemeval",
        # "chunks_test_longmemeval",
        # "memories_eval_longmemeval",
        # "facts_eval_longmemeval",
        # "chunks_eval_longmemeval",
        # "memories_longmemeval",
        # "facts_longmemeval",
        # "chunks_longmemeval",
        # "facts_test_longmemeval_v1",
        # "memories_test_longmemeval_v1",
        # "chunks_test_longmemeval_v1",
        # "facts_longmemeval_v2",
        # "memories_longmemeval_v2",
        # "chunks_longmemeval_v2",
        # "chunks_test_longmemeval_v2",
        # "facts_test_longmemeval_v2",
        # "memories_test_longmemeval_v2",
        # "chunks_longmemeval_v1",
        # # 其他可能的集合名称
        # "memories",
        # "facts",
        # "chunks"
        "memories_test_longmemeval_fmc",
        "facts_test_longmemeval_fmc",
        "chunks_test_longmemeval_fmc",
    ]
    
    # 删除所有相关集合
    for collection_name in collections_to_delete:
        delete_collection(client, collection_name)
    
    print("\n🎉 数据库清空完成！")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
