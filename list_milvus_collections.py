#!/usr/bin/env python3
"""
列出Milvus数据库中的所有集合

该脚本用于连接到Milvus数据库并列出所有集合名称
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

# 导入pymilvus库
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


def list_collections():
    """列出Milvus数据库中的所有集合"""
    print(f"\n📋 开始列出Milvus数据库中的集合...")
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
    
    # 列出所有集合
    try:
        collections = client.list_collections()
        print(f"\n📊 数据库中共有 {len(collections)} 个集合:")
        for i, collection in enumerate(collections, 1):
            print(f"   {i}. {collection}")
        
        # 获取每个集合的统计信息（可选）
        print("\n📈 集合统计信息:")
        for collection in collections:
            try:
                stats = client.get_collection_stats(collection)
                print(f"   {collection}: {stats}")
            except Exception as e:
                print(f"   {collection}: 获取统计信息失败 - {e}")
        
    except Exception as e:
        print(f"❌ 列出集合失败: {e}")
        return 1
    
    print("\n🎉 列出集合完成！")
    return 0


def main():
    """主函数"""
    return list_collections()


if __name__ == "__main__":
    import sys
    sys.exit(main())
