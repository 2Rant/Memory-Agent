import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class VectorDBConfig:
    """向量数据库配置类"""
    uri: str
    user_name: str = ""
    password: str = ""
    db_name: str = "default"
    dimension: int = 1536
    vector_db_type: str = "milvus"  # 支持 "milvus" 或 "qdrant"
    api_key: str = ""  # 用于Qdrant的API密钥

class VectorDBInterface(ABC):
    """向量数据库抽象接口"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
    
    @abstractmethod
    def create_collection(self, name: str, schema: Any):
        """创建集合"""
        pass
    
    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """检查集合是否存在"""
        pass
    
    @abstractmethod
    def drop_collection(self, name: str):
        """删除集合"""
        pass
    
    @abstractmethod
    def insert(self, collection_name: str, rows: List[Dict]):
        """插入数据"""
        pass
    
    @abstractmethod
    def upsert(self, collection_name: str, rows: List[Dict]):
        """更新或插入数据"""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], filter: str = "", limit: int = 5, output_fields: List[str] = None, similarity_threshold: Optional[float] = None):
        """搜索向量"""
        pass
    
    @abstractmethod
    def search_bm25(self, collection_name: str, query_text: str, filter: str = "", limit: int = 5, output_fields: List[str] = None):
        """BM25 关键词搜索"""
        pass

    @abstractmethod
    def query(self, collection_name: str, filter: str, output_fields: List[str], limit: int = 100):
        """查询数据"""
        pass
    
    @abstractmethod
    def load_collection(self, name: str):
        """加载集合"""
        pass
    
    @abstractmethod
    def create_index(self, collection_name: str, index_params: Any):
        """创建索引"""
        pass
    
    @abstractmethod
    def prepare_index_params(self):
        """准备索引参数"""
        pass
    
    @abstractmethod
    def create_schema(self, auto_id: bool = False, enable_dynamic_field: bool = True):
        """创建 schema"""
        pass

class MilvusDB(VectorDBInterface):
    """Milvus 向量数据库实现"""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        from pymilvus import MilvusClient, DataType, exceptions
        self.client = MilvusClient(uri=config.uri, user=config.user_name, password=config.password, db_name=config.db_name)
        self.DataType = DataType
        self.exceptions = exceptions
    
    def create_collection(self, name: str, schema: Any, index_params: Any = None):
        if index_params:
            return self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        return self.client.create_collection(collection_name=name, schema=schema)
    
    def has_collection(self, name: str) -> bool:
        return self.client.has_collection(name)
    
    def drop_collection(self, name: str):
        return self.client.drop_collection(name)
    
    def insert(self, collection_name: str, rows: List[Dict]):
        return self.client.insert(collection_name=collection_name, data=rows)
    
    def upsert(self, collection_name: str, rows: List[Dict]):
        return self.client.upsert(collection_name=collection_name, data=rows)
    
    def search(self, collection_name: str, query_vector: List[float], filter: str = "", limit: int = 5, output_fields: List[str] = None, similarity_threshold: float = None):
        if output_fields is None:
            output_fields = []
        
        # 处理 query_vector，支持任意深度嵌套列表的情况
        actual_query_vector = query_vector
        # 循环处理，直到 actual_query_vector 不再是嵌套列表
        while isinstance(actual_query_vector, list) and len(actual_query_vector) > 0 and isinstance(actual_query_vector[0], list):
            actual_query_vector = actual_query_vector[0]
        
        # 调用 Milvus 客户端搜索
        results = self.client.search(
            collection_name=collection_name,
            data=[actual_query_vector],
            filter=filter,
            limit=limit,
            output_fields=output_fields
        )
        
        # 应用相似度阈值过滤
        if similarity_threshold is not None and results and results[0]:
            # 注意：Milvus 返回的是距离值，余弦相似度中距离越小相似度越高
            # 转换为相似度分数：相似度 = 1 - 距离
            filtered_results = []
            for hit in results[0]:
                distance = hit['distance']
                similarity = 1 - distance
                if similarity >= similarity_threshold:
                    filtered_results.append(hit)
            
            # 替换为过滤后的结果
            results[0] = filtered_results
        
        return results
    
    def search_bm25(self, collection_name: str, query_text: str, filter: str = "", limit: int = 5, output_fields: List[str] = None):
        """Milvus BM25 搜索实现 (需要 Milvus 2.4+ 支持全文检索)"""
        # 注意：这里假设已经创建了支持全文检索的 Function 字段和 Index
        # 由于 Milvus Python SDK 的 search 方法直接支持 bm25 function call，这里需要根据具体 SDK 版本适配
        # 简单实现：使用 search 方法，传入 bm25 参数
        # 实际情况可能需要 Function 对象，这里暂时模拟抛出未实现，因为 Milvus 的 BM25 设置比较复杂且版本依赖强
        # 如果需要实现，通常需要先在 schema 定义时启用 enable_dynamic_field 和创建 function
        
        # 这里的实现是一个占位符，因为当前环境可能不支持 Milvus 2.4 的高级特性
        # 实际项目中建议使用单独的 ES 或 Tantivy 来做 BM25，或者确认 Milvus 版本
        print("⚠️ MilvusDB.search_bm25 尚未完整实现，返回空结果")
        return [[]]

    def query(self, collection_name: str, filter: str, output_fields: List[str], limit: int = 100):
        return self.client.query(
            collection_name=collection_name,
            filter=filter,
            output_fields=output_fields,
            limit=limit
        )
    
    def load_collection(self, name: str):
        return self.client.load_collection(name)
    
    def create_index(self, collection_name: str, index_params: Any):
        return self.client.create_index(collection_name=collection_name, index_params=index_params)
    
    def prepare_index_params(self):
        return self.client.prepare_index_params()
    
    def create_schema(self, auto_id: bool = False, enable_dynamic_field: bool = True):
        return self.client.create_schema(auto_id=auto_id, enable_dynamic_field=enable_dynamic_field)

class QdrantDB(VectorDBInterface):
    """Qdrant 向量数据库实现"""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        # 使用配置中的api_key，如果没有则尝试从环境变量获取
        api_key = config.api_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=config.uri, api_key=api_key)
        self.VectorParams = VectorParams
        self.Distance = Distance
        self.PointStruct = PointStruct
    
    def create_collection(self, name: str, schema: Any = None):
        # Qdrant 使用不同的方式创建集合，不需要 schema
        # 使用 VectorParams 来配置向量字段
        return self.client.create_collection(
            collection_name=name,
            vectors_config=self.VectorParams(size=self.config.dimension, distance=self.Distance.DOT)
        )
    
    def has_collection(self, name: str) -> bool:
        return self.client.collection_exists(collection_name=name)
    
    def drop_collection(self, name: str):
        return self.client.delete_collection(collection_name=name)
    
    def insert(self, collection_name: str, rows: List[Dict]):
        # 转换为 Qdrant 的 PointStruct 格式
        points = []
        for row in rows:
            point_id = row.get("memory_id") or row.get("fact_id") or row.get("chunk_id")
            # 生成 UUID 如果没有 ID
            import uuid
            if not point_id:
                point_id = str(uuid.uuid4())
            
            # 提取向量字段
            vector = row.get("embedding") or row.get("dummy_embedding") or [0.0] * self.config.dimension
            
            # 提取 payload
            payload = row.copy()
            if "embedding" in payload:
                del payload["embedding"]
            if "dummy_embedding" in payload:
                del payload["dummy_embedding"]
            
            points.append(self.PointStruct(id=point_id, vector=vector, payload=payload))
        
        return self.client.upsert(collection_name=collection_name, points=points)
    
    def upsert(self, collection_name: str, rows: List[Dict]):
        # Qdrant 只有 upsert 方法，没有单独的 insert 方法
        return self.insert(collection_name, rows)
    
    def search(self, collection_name: str, query_vector: List[float], filter: str = "", limit: int = 5, output_fields: List[str] = None, similarity_threshold: float = None):
        from qdrant_client.models import Filter, MatchValue, FieldCondition
        
        qdrant_filter = None
        if filter:
            # 简单的过滤表达式转换，仅支持基本的等式过滤
            try:
                if "==" in filter:
                    key, value = filter.split("==")
                    key = key.strip()
                    value = value.strip().strip("'\"\n")
                    # 处理布尔值
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    # 尝试转换为整数
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    qdrant_filter = Filter(
                        must=[FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )]
                    )
            except Exception as e:
                print(f"无法解析 Qdrant 过滤表达式: {e}")
        
        # 处理 query_vector，支持任意深度嵌套列表的情况
        actual_query_vector = query_vector
        # 循环处理，直到 actual_query_vector 不再是嵌套列表
        while isinstance(actual_query_vector, list) and len(actual_query_vector) > 0 and isinstance(actual_query_vector[0], list):
            actual_query_vector = actual_query_vector[0]
        
        # 使用 query 方法代替 search，Qdrant v1.16+ 使用 query 方法
        from qdrant_client.models import VectorParams, Distance
        results = self.client.query(
            collection_name=collection_name,
            query_vector=actual_query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        # 转换为与 Milvus 兼容的格式
        formatted_results = []
        for result in results:
            entity = result.payload
            entity["distance"] = result.score
            formatted_results.append({"entity": entity, "distance": result.score})
        
        # 应用相似度阈值过滤
        if similarity_threshold is not None:
            filtered_results = []
            for hit in formatted_results:
                # Qdrant 返回的是相似度分数，分数越高相似度越高
                if hit['distance'] >= similarity_threshold:
                    filtered_results.append(hit)
            return [filtered_results]
        
        return [formatted_results]
    
    def search_bm25(self, collection_name: str, query_text: str, filter: str = "", limit: int = 5, output_fields: List[str] = None):
        """Qdrant BM25 搜索实现 (基于 Qdrant 的全文检索支持)"""
        # Qdrant 支持 payload 的全文索引 (full-text index)
        # 这里使用 scroll API 配合 filter 来模拟关键词搜索
        from qdrant_client.models import Filter, MatchValue, FieldCondition, MatchText
        
        qdrant_filter = None
        # 解析基础 filter (如 user_id)
        must_conditions = []
        if filter:
             try:
                if "==" in filter:
                    key, value = filter.split("==")
                    key = key.strip()
                    value = value.strip().strip("'\"\n")
                    must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
             except:
                 pass
        
        # 添加文本搜索条件
        # 假设我们搜索 'content' 字段 (对于记忆) 或 'text' 字段 (对于事实)
        # 这里需要知道搜索哪个字段，通常 output_fields 包含该字段，或者默认搜索 text/content
        search_field = "content" 
        # 简单的启发式：如果 collection 名字包含 fact，搜索 text
        if "fact" in collection_name:
            search_field = "text"
            
        must_conditions.append(FieldCondition(key=search_field, match=MatchText(text=query_text)))
        
        qdrant_filter = Filter(must=must_conditions)
        
        # 使用 scroll 接口获取匹配项
        # 注意：Qdrant 的 scroll 不会按 BM25 分数排序，它主要用于过滤
        # 要实现真正的 BM25 排序，Qdrant 需要使用专门的发现/推荐 API 或者客户端重排
        # 这里作为一个简化的关键词匹配实现
        results, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        formatted_results = []
        for point in results:
            entity = point.payload
            # BM25 分数这里无法直接获取，暂时给一个默认高分或者基于匹配程度的伪分数
            # 这是一个简化的实现
            formatted_results.append({"entity": entity, "distance": 1.0}) # 1.0 表示关键词匹配命中
            
        return [formatted_results]

    def query(self, collection_name: str, filter: str, output_fields: List[str] = None, limit: int = 100):
        # Qdrant 的 query_points 方法不支持复杂的过滤表达式
        # 这里使用 search 方法模拟 query 功能
        # 使用一个全零向量作为查询向量
        query_vector = [0.0] * self.config.dimension
        results = self.search(collection_name, query_vector, filter, limit, output_fields)
        
        # 提取实体数据
        entities = []
        for result in results[0]:
            entities.append(result["entity"])
        
        return entities
    
    def load_collection(self, name: str):
        # Qdrant 不需要显式加载集合
        pass
    
    def create_index(self, collection_name: str, index_params: Any):
        # Qdrant 会自动创建索引
        pass
    
    def prepare_index_params(self):
        # Qdrant 不需要准备索引参数
        return None
    
    def create_schema(self, auto_id: bool = False, enable_dynamic_field: bool = True):
        # Qdrant 不需要 schema
        return None

class VectorDBFactory:
    """向量数据库工厂类，用于生成不同类型的向量数据库客户端"""
    
    @staticmethod
    def create_db(config: VectorDBConfig) -> VectorDBInterface:
        """创建向量数据库客户端"""
        if config.vector_db_type == "milvus":
            return MilvusDB(config)
        elif config.vector_db_type == "qdrant":
            return QdrantDB(config)
        else:
            raise ValueError(f"不支持的向量数据库类型: {config.vector_db_type}")
