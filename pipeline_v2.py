import os
import time
import uuid
import json
import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from utils import (MEMREADER_PROMPT, 
                   get_embedding, parse_messages, LME_JUDGE_MODEL_TEMPLATE, 
                   LME_ANSWER_PROMPT, remove_code_blocks, extract_json)
from lme_eval import lme_grader
from datetime import datetime, timezone
import pytz
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from vector_db import VectorDBConfig, VectorDBFactory
# ==========================================
# 0. Setup & Prompts
# ==========================================
load_dotenv()

# ⚠️ 请确保环境变量中有 OPENAI_API_KEY 和 MILVUS_URI
# 如果是本地测试，确保 Docker 中 Milvus 已启动

llm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

MEMORY_MANAGER_PROMPT = """You are a specialized Memory Manager Agent.
Your role is to maintain the consistency and growth of a memory graph using the provided tools.

[INPUTS]
You will receive:
1. "New Facts": A list of atomic facts extracted from the latest user input.
2. "Existing Memories": A list of retrieved memory items, each with a simplified Integer ID (e.g., "0", "1", "2").
   - These memories include those directly related to the new facts, as well as other related facts connected with these memories.
   - They form a connected graph of information relevant to the new facts.
3. "Existing Facts": A list of retrieved fact items, each with a simplified Integer ID (e.g., "F0", "F1", "F2").
   - These facts include those directly related to the new facts, with their text and details.
   - They provide additional context and historical information about the new facts.

[MANDATORY OUTPUT FORMAT]
For every new fact you process, you MUST:
1. First generate a detailed thinking process
2. Then call the appropriate tool

[THINKING PROCESS REQUIREMENTS]
Your thinking process MUST include:
- The specific new fact you're analyzing
- Which existing memories are relevant (with their IDs)
- Which existing facts are relevant (with their IDs)
- How memories and facts are connected
- Your comparison and reasoning
- Which operation you've decided to perform and why

[OPERATIONS & GUIDELINES]
Compare New Facts with Existing Memories and Existing Facts, then perform the following operations using the available tools. 
DO NOT output raw JSON text. You MUST use the provided function tools.

1. **ADD MEMORY (create_memory)**
   - **Condition**: If a fact contains completely NEW information not present in Existing Memories or Existing Facts.
   - **Action**: Call `create_memory` with a concise summary of the facts, not just a simple concatenation.
   - **Important**: Memory content should be a meaningful and concise summary.
  
2. **UPDATE MEMORY (update_memory)**
   - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Memory.
   - **Constraint**: You MUST use the Integer ID (e.g., "0") provided in the input as the `target_memory_id`.
   - **Logic**: Merge the old content and new fact into a comprehensive statement, not just a simple concatenation.
   - **Example**:
     - Old (ID="0"): "User likes generic pizza."
     - New Fact: "User loves pepperoni pizza."
     - Action: `update_memory(target_memory_id="0", new_content="User loves pepperoni pizza", ...)`

3. **DELETE MEMORY (delete_memory)**
   - **Condition**: If a fact explicitly contradicts an Existing Memory (and the new fact is trusted), or if the memory is no longer valid.
   - **Constraint**: Use the Integer ID (e.g., "1") as `target_memory_id`.

4. **INFER MEMORY (infer_memory)**
   - **Condition**: Look for higher-level insights. If combining "Memory A" and "Memory B" reveals a hidden connection or causality.
   - **Action**: Call `infer_memory`.
   - **Example**:
     - Memory A (ID="2"): "User moved to Singapore."
     - Memory B (ID="3"): "User bought a Type G power adapter."
     - Inference: "User is preparing electronics for Singapore power standards."
     - Action: `infer_memory(source_memory_ids=["2", "3"], inference_content="...")`

5. **ADD FACT (fact_add)**
   - **Condition**: If a new fact is completely unrelated to all Existing Facts.
   - **Action**: Call `fact_add` to save the new fact to the database.
   - **Example**:
     - New Fact: "User just bought a new laptop."
     - Details: ["brand: Apple", "model: MacBook Pro", "purchase_date: 2023-10-15"]
     - No related Existing Facts found.
     - Action: `fact_add(content="User just bought a new laptop", details=["brand: Apple", "model: MacBook Pro", "purchase_date: 2023-10-15"])`

6. **UPDATE FACT (fact_trajectorize)**
   - **Condition**: If a new fact is related to an Existing Fact (e.g., updates, corrects, or expands it).
   - **Action**: Call `fact_trajectorize` to update the Existing Fact to the new content while recording the change trajectory.
   - **Constraint**: You MUST use the Integer ID (e.g., "F0") provided in the input as the `target_fact_id`.
   - **Example**:
     - Old Fact (ID="F0"): "User lives in Beijing."
     - Old Details: ["city: Beijing", "duration: 5 years"]
     - New Fact: "User moved to Shanghai."
     - New Details: ["city: Shanghai", "duration: 0 months"]
     - Action: `fact_trajectorize(target_fact_id="F0", new_content="User moved to Shanghai", diff="Changed residence from Beijing to Shanghai", details=["city: Shanghai", "duration: 0 months"])`

7. **NOOP (no_operation)**
   - **Condition**: If the fact is redundant (already exactly covered by memory or fact) or trivial.
"""

# --- TOOLS ---
MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_memory",
            "description": "Create a NEW independent memory node with a concise summary of the facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The concise summary content of the new memory, not just a list of facts."},
                    "evidence_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts supporting this memory."}
                },
                "required": ["content", "evidence_facts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update an existing memory by merging the old content and new fact into a comprehensive, concise statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_memory_id": {"type": "string", "description": "The simplified Integer ID (e.g., '0') of the memory to update, found in the [EXISTING MEMORIES] list."},
                    "new_content": {"type": "string", "description": "The merged/updated comprehensive statement."},
                    "evidence_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts supporting this update."}
                },
                "required": ["target_memory_id", "new_content", "evidence_facts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "infer_memory",
            "description": "Look for higher-level insights. If combining multiple existing memories reveals a hidden connection or causality, create an inferred memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_memory_ids": {"type": "array", "items": {"type": "string"}, "description": "List of simplified Integer IDs (e.g., ['0', '1']) acting as premises, found in the [EXISTING MEMORIES] list."},
                    "inference_content": {"type": "string", "description": "The higher-level insight or inference derived from combining the source memories."},
                    "evidence_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts supporting this inference."}
                },
                "required": ["source_memory_ids", "inference_content", "evidence_facts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Archive/Soft-delete a memory if it explicitly contradicts a new fact (and the new fact is trusted), or if the memory is no longer valid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_memory_id": {"type": "string", "description": "The simplified Integer ID (e.g., '1') of the memory to delete, found in the [EXISTING MEMORIES] list."},
                    "evidence_facts": {"type": "array", "items": {"type": "string"}, "description": "Facts supporting this deletion."}
                },
                "required": ["target_memory_id", "evidence_facts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_add",
            "description": "Add a new fact to the database when it's completely unrelated to all existing facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content of the new fact."},
                    "details": {"type": "array", "items": {"type": "string"}, "description": "Additional details about the fact, as strings in format 'key: value'."}
                },
                "required": ["content", "details"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fact_trajectorize",
            "description": "Update an existing fact with new content while recording the change trajectory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_fact_id": {"type": "string", "description": "The simplified Integer ID (e.g., 'F0') of the fact to update, found in the [EXISTING FACTS] list."},
                    "new_content": {"type": "string", "description": "The new content for the fact."},
                    "diff": {"type": "string", "description": "Description of the difference between old and new content."},
                    "details": {"type": "array", "items": {"type": "string"}, "description": "Additional details about the fact, as strings in format 'key: value'."}
                },
                "required": ["target_fact_id", "new_content", "diff", "details"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "no_operation",
            "description": "No action needed if the fact is redundant (already exactly covered by memory or fact) or trivial.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "The reason for no operation."}},
                "required": ["reason"]
            }
        }
    }
]

# --- UTILS ---
def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    return llm_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

@dataclass
class MilvusConfig:
    """Milvus配置类（兼容旧代码）"""
    uri: str = os.getenv("MILVUS_URI")
    user_name: str = os.getenv("MILVUS_USER_NAME")
    # password: str = os.getenv("MILVUS_PASSWORD")
    db_name: str = os.getenv("MILVUS_DB_NAME", "default")
    dimension: int = 1536
    
    def to_vector_db_config(self, vector_db_type: str = "milvus") -> VectorDBConfig:
        """转换为VectorDBConfig"""
        # 确保vector_db_type是字符串类型
        if not isinstance(vector_db_type, str):
            vector_db_type = "milvus"  # 默认使用milvus
        
        # 根据vector_db_type选择不同的URL
        if vector_db_type == "qdrant":
            uri = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            user_name = ""
            password = ""
        else:
            uri = self.uri
            api_key = ""
            user_name = self.user_name
            password = os.getenv("MILVUS_PASSWORD")
        
        return VectorDBConfig(
            uri=uri,
            user_name=user_name,
            password=password,
            api_key=api_key,
            db_name=self.db_name,
            dimension=self.dimension,
            vector_db_type=vector_db_type
        )

# ==========================================
# 1. Pipeline Class
# ==========================================
class MemoryPipeline:
    def __init__(self, config=None, vector_db_type="milvus", clear_db=False, mode='eval', dataset_name=""):
        """
        初始化MemoryPipeline
        
        Args:
            config: MilvusConfig或VectorDBConfig实例，如果为None则使用默认配置
            vector_db_type: 指定使用的向量数据库类型，支持"milvus"或"qdrant"
            clear_db: 是否清空数据库，默认为False
            dataset_name: 数据集名称，用于集合名称后缀，默认为空
        """
        # 如果没有提供配置，创建默认配置
        if config is None:
            config = MilvusConfig()
        
        self.config = config
        
        # 转换为VectorDBConfig
        if hasattr(config, 'to_vector_db_config'):
            vector_db_config = config.to_vector_db_config(vector_db_type=vector_db_type)
        else:
            # 如果已经是VectorDBConfig实例，直接使用
            vector_db_config = config
        
        # 使用工厂类创建向量数据库客户端
        self.client = VectorDBFactory.create_db(vector_db_config)
        
        # 根据模式和数据集名称设置集合名称
        base_suffix = "_test" if mode == 'test' else ""
        dataset_suffix = f"_{dataset_name}" if dataset_name else ""
        full_suffix = f"{base_suffix}{dataset_suffix}"
        
        self.mem_col = f"memories{full_suffix}"
        self.fact_col = f"facts{full_suffix}"
        self.chunk_col = f"chunks{full_suffix}"
        
        self.dim = vector_db_config.dimension  # Save dimension as instance variable
        # 初始化操作次数计数器
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        self._init_collections(clear_db=clear_db)

    def _init_collections(self, clear_db=False):
        dim = self.config.dimension
        
        # 如果需要清空数据库，先删除所有集合
        if clear_db:
            print("正在清空数据库...")
            # 直接删除集合，不检查存在性
            self.client.drop_collection(self.mem_col)
            self.client.drop_collection(self.fact_col)
            self.client.drop_collection(self.chunk_col)
            print("数据库清空完成.")
        
        # 直接创建或获取集合，不进行存在性检查
        # 创建集合的逻辑已经包含了如果集合存在则跳过的处理
        
        # 处理 memories 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端，创建完整的schema
            s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            s.add_field("memory_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
            s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
            s.add_field("content", self.client.DataType.VARCHAR, max_length=65535)
            s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)
            s.add_field("status", self.client.DataType.VARCHAR, max_length=16)
            s.add_field("created_at", self.client.DataType.INT64)
            s.add_field("updated_at", self.client.DataType.INT64)
            s.add_field("relations", self.client.DataType.JSON) 
            
            # 创建集合
            self.client.create_collection(self.mem_col, schema=s)
            print(f"Collection '{self.mem_col}' created or exists.")
            
            # 直接创建索引，不检查索引是否存在
            # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
            try:
                print(f"为集合 '{self.mem_col}' 创建索引...")
                idx_params = self.client.prepare_index_params()
                idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                self.client.create_index(self.mem_col, index_params=idx_params)
                print(f"集合 '{self.mem_col}' 的索引创建成功或已存在")
            except Exception as e:
                print(f"创建索引失败: {e}")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.mem_col)
            print(f"Collection '{self.mem_col}' created or exists.")
        
        # 处理 facts 集合
        if hasattr(self.client, 'DataType'):
            s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            s.add_field("fact_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
            s.add_field("linked_memory_ids", self.client.DataType.JSON)
            s.add_field("linked_chunk_id", self.client.DataType.VARCHAR, max_length=64)
            s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
            s.add_field("details", self.client.DataType.JSON)  # 添加details字段
            s.add_field("timestamp", self.client.DataType.INT64)
            s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)  # 添加user_id字段
            s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
            s.add_field("trajectory", self.client.DataType.JSON, default=[])  # 添加trajectory字段，记录变化轨迹
            
            # 创建集合
            self.client.create_collection(self.fact_col, schema=s)
            print(f"Collection '{self.fact_col}' created or exists.")
            
            # 直接创建索引，不检查索引是否存在
            # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
            try:
                print(f"为集合 '{self.fact_col}' 创建索引...")
                idx_params = self.client.prepare_index_params()
                idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                self.client.create_index(self.fact_col, index_params=idx_params)
                print(f"集合 '{self.fact_col}' 的索引创建成功或已存在")
            except Exception as e:
                print(f"创建索引失败: {e}")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.fact_col)
            print(f"Collection '{self.fact_col}' created or exists.")
        
        # 处理 chunks 集合
        if hasattr(self.client, 'DataType'):
            s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            s.add_field("chunk_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
            s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
            s.add_field("timestamp", self.client.DataType.INT64)
            s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
            
            # 创建集合
            self.client.create_collection(self.chunk_col, schema=s)
            print(f"Collection '{self.chunk_col}' created or exists.")
            
            # 直接创建索引，不检查索引是否存在
            # Milvus的create_index方法会在索引已存在时自动跳过或返回成功
            try:
                print(f"为集合 '{self.chunk_col}' 创建索引...")
                idx_params = self.client.prepare_index_params()
                idx_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
                self.client.create_index(self.chunk_col, index_params=idx_params)
                print(f"集合 '{self.chunk_col}' 的索引创建成功或已存在")
            except Exception as e:
                print(f"创建索引失败: {e}")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.chunk_col)
            print(f"Collection '{self.chunk_col}' created or exists.")
        
        # 直接加载所有集合，不进行复杂的错误处理
        print("Loading collections into memory...")
        
        # 加载集合（Qdrant 不需要显式加载）
        if hasattr(self.client, 'load_collection'):
            # 为每个集合创建索引后直接加载
            print(f"加载集合 '{self.mem_col}'...")
            self.client.load_collection(self.mem_col)
            
            print(f"加载集合 '{self.fact_col}'...")
            self.client.load_collection(self.fact_col)
            
            print(f"加载集合 '{self.chunk_col}'...")
            self.client.load_collection(self.chunk_col)
            
            print("All collections loaded successfully.")

    # --- Step 1: Extract ---
    def step_extract(self, chunk_text: str, extract_mode: str = "whole") -> Dict:
        """
        从对话中提取事实
        
        Args:
            chunk_text: 对话文本
            extract_mode: 提取模式，可选值：
                - "whole": 对整个chunk进行提取
                - "turn": 按轮次提取，每轮user-assistant对话单独提取
        
        Returns:
            包含提取事实的字典
        """
        # print(f"\n👀 [1. Extract] Processing: '{chunk_text}'")
        
        # 如果是按轮次提取，先解析对话轮次
        if extract_mode == "turn":
            # 尝试解析对话轮次
            try:
                # 简单的轮次检测：查找user:和assistant:的组合
                import re
                # 匹配user: ... assistant: ... 的模式
                turn_pattern = r'(user: .*?)(?=assistant: |$)' 
                turns = re.findall(turn_pattern, chunk_text, re.DOTALL)
                
                # 如果找到轮次，单独处理每轮
                if turns:
                    all_facts = []
                    for turn in turns:
                        # 确保每轮都有完整的user-assistant对话
                        turn_text = turn.strip()
                        if turn_text:
                            # 对单轮对话提取事实
                            turn_facts = self._extract_single_turn(turn_text)
                            all_facts.extend(turn_facts)
                    
                    return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": all_facts, "timestamp": int(time.time())}
            except Exception as e:
                print(f"解析对话轮次失败，回退到whole模式: {e}")
        
        # 默认模式：对整个chunk进行提取
        facts = self._extract_single_turn(chunk_text)
        return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": facts, "timestamp": int(time.time())}
    
    def _extract_single_turn(self, text: str) -> List[Dict]:
        """
        对单个文本片段提取事实
        
        Args:
            text: 要提取事实的文本
            
        Returns:
            提取到的事实列表
        """
        try:
            response = llm_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                        {"role": "system", "content": MEMREADER_PROMPT}, 
                        {"role": "user", "content": text}],
                response_format={"type": "json_object"}, temperature=0
            )
            fact_objects = json.loads(response.choices[0].message.content).get("facts", [])
            # 保留完整的fact对象，包括details信息
            facts = []
            for fact_obj in fact_objects:
                if fact_obj.get("fact"):
                    facts.append({
                        "text": fact_obj.get("fact", ""),
                        "details": fact_obj.get("details", [])
                    })
        except Exception as e: 
            print(f"Extraction failed: {e}")
            facts = [{"text": text, "details": []}]
        return facts

    # --- Step 2: Retrieve ---    
    def step_retrieve(self, extract_result: Dict, limit: int = 3, user_id: str = 'default', similarity_threshold: float = None) -> List[Dict]:
        new_facts = extract_result['new_facts']
        if not new_facts: return []
        
        print(f"🔍 [2. Retrieve] Searching Memories and Facts for {len(new_facts)} facts...")
        context_bundles = []

        for fact in new_facts:
            query_vec = get_embedding(fact['text'])
            
            # 1. 搜索相关的memories
            print(f"   Searching memories for fact: {fact['text'][:50]}...")
            mem_res = self.client.search(
                self.mem_col, [query_vec], filter=f"status == 'active' and user_id == '{user_id}'", limit=limit,
                output_fields=["content", "memory_id", "created_at"],
                similarity_threshold=similarity_threshold
            )
            
            candidates = []
            if mem_res and mem_res[0]:
                for hit in mem_res[0]:
                    candidates.append(hit['entity'])
            
            # 2. 搜索相关的facts
            print(f"   Searching facts for fact: {fact['text'][:50]}...")
            fact_res = self.client.search(
                self.fact_col, [query_vec], filter=f"user_id == '{user_id}'", limit=limit * 2,
                output_fields=["fact_id", "text", "details", "timestamp", "linked_memory_ids", "trajectory"],
                similarity_threshold=similarity_threshold
            )
            
            related_facts = []
            if fact_res and fact_res[0]:
                for hit in fact_res[0]:
                    related_facts.append(hit['entity'])
            
            # 3. 检索记忆关联的事实
            if candidates:
                memory_ids = [mem['memory_id'] for mem in candidates]
                expr_parts = [f'array_contains(linked_memory_ids, "{mem_id}")' for mem_id in memory_ids]
                filter_expr = " || ".join(expr_parts)
                
                try:
                    mem_related_facts = self.client.query(
                        collection_name=self.fact_col,
                        filter=filter_expr,
                        output_fields=["fact_id", "linked_memory_ids", "text", "linked_chunk_id", "timestamp", "details", "trajectory"]
                    )
                    # 合并相关事实，去重
                    seen_fact_ids = set(f['fact_id'] for f in related_facts)
                    for f in mem_related_facts:
                        if f['fact_id'] not in seen_fact_ids:
                            seen_fact_ids.add(f['fact_id'])
                            related_facts.append(f)
                except Exception as e:
                    print(f"   ⚠️ Error retrieving memory-related facts: {e}")
            
            # 4. 检索这些事实关联的其他记忆
            if related_facts:
                all_related_memory_ids = set()
                for f in related_facts:
                    linked_mem_ids = f.get("linked_memory_ids", [])
                    all_related_memory_ids.update(linked_mem_ids)
                
                existing_memory_ids = set([mem['memory_id'] for mem in candidates])
                new_memory_ids = all_related_memory_ids - existing_memory_ids
                
                if new_memory_ids:
                    quoted_new_ids = [f'"{mem_id}"' for mem_id in new_memory_ids]
                    mem_filter = f"status == 'active' and user_id == '{user_id}' and memory_id in [{','.join(quoted_new_ids)}]"
                    
                    try:
                        additional_memories = self.client.query(
                            collection_name=self.mem_col,
                            filter=mem_filter,
                            output_fields=["content", "memory_id", "created_at"]
                        )
                        candidates.extend(additional_memories)
                    except Exception as e:
                        print(f"   ⚠️ Error retrieving additional memories: {e}")
            
            # 5. 对候选记忆进行去重
            unique_candidates = []
            seen_memory_ids = set()
            for mem in candidates:
                mem_id = mem['memory_id']
                if mem_id not in seen_memory_ids:
                    seen_memory_ids.add(mem_id)
                    unique_candidates.append(mem)
            
            # 6. 对相关事实进行去重
            unique_related_facts = []
            seen_fact_ids = set()
            for f in related_facts:
                fact_id = f['fact_id']
                if fact_id not in seen_fact_ids:
                    seen_fact_ids.add(fact_id)
                    unique_related_facts.append(f)
            
            context_bundles.append({
                "new_fact": fact,
                "candidates": unique_candidates,
                "related_facts": unique_related_facts
            })
            
        return context_bundles

    # --- Step 3: Decide (With ID Mapping) ---
    def step_decide(self, extract_result: Dict, context_bundles: List[Dict], user_id: str = 'default', training_mode: bool = False) -> List[Dict]:
        all_new_facts = extract_result['new_facts']
        
        # 1. 合并去重 Candidates (Memories)
        temp_mem_storage = {}
        for bundle in context_bundles:
            for mem in bundle['candidates']:
                temp_mem_storage[mem['memory_id']] = mem
        
        unique_memories_list = list(temp_mem_storage.values())
        
        # 2. 合并去重 Related Facts
        temp_fact_storage = {}
        for bundle in context_bundles:
            if 'related_facts' in bundle:
                for fact in bundle['related_facts']:
                    temp_fact_storage[fact['fact_id']] = fact
        
        unique_facts_list = list(temp_fact_storage.values())
        
        if not training_mode:
            print(f"🧠 [3. Manager] Global Decide: {len(all_new_facts)} facts vs {len(unique_memories_list)} memories vs {len(unique_facts_list)} facts.")

        # 🌟 3. 构造 ID 映射 (Mapping Logic)
        uuid_mapping = {}  # { "0": "real-uuid", "1": "real-uuid" }
        fact_id_mapping = {}  # { "F0": "real-fact-id", "F1": "real-fact-id" }
        
        candidates_str = ""
        facts_str = ""

        # 构造Memories部分
        if not unique_memories_list:
            candidates_str = "(No relevant memories found. Treat as new topic.)"
        else:
            for idx, mem in enumerate(unique_memories_list):
                simple_id = str(idx)
                real_uuid = mem['memory_id']
                uuid_mapping[simple_id] = real_uuid
                candidates_str += f"[Memory Item ID: {simple_id}]\n- Content: {mem['content']}\n\n"
        
        # 构造Facts部分
        if not unique_facts_list:
            facts_str = "(No relevant facts found.)"
        else:
            for idx, fact in enumerate(unique_facts_list):
                simple_id = f"F{idx}"
                real_fact_id = fact['fact_id']
                fact_id_mapping[simple_id] = real_fact_id
                facts_str += f"[Fact Item ID: {simple_id}]\n- Text: {fact['text']}\n- Details: {json.dumps(fact.get('details', []), ensure_ascii=False)}\n\n"

        # 构造最终 Prompt
        system_msg = MEMORY_MANAGER_PROMPT

        # 构造包含text、details和fact_index的fact字符串
        fact_objects = []
        for idx, fact in enumerate(all_new_facts):
            fact_obj = {
                "text": fact['text'], 
                "details": fact.get('details', []),
                "fact_index": idx  # 添加fact_index，用于标识同一事实
            }
            fact_objects.append(fact_obj)
        
        user_content = f"""
        [New Facts Stream]
        {json.dumps(fact_objects, ensure_ascii=False, indent=2)}
        
        [EXISTING MEMORIES]
        {candidates_str}
        
        [EXISTING FACTS]
        {facts_str}
        """

        all_decisions = []
        try:
            # 使用streaming模式来获取完整的响应，包括思维过程
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                tools=MEMORY_TOOLS,
                tool_choice="required",
                temperature=0,
                stream=True
            )
            
            # 收集完整的响应
            collected_messages = []
            for chunk in response:
                try:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content is not None:
                            collected_messages.append(choice.delta.content)
                except IndexError:
                    continue
            
            # 拼接完整的思考过程
            thinking_process = ''.join(collected_messages)
            if thinking_process and not training_mode:
                print(f"\n   🧠 LLM思考过程:")
                print(f"   {thinking_process}")
            
            # 重新创建非流式响应以获取工具调用
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                tools=MEMORY_TOOLS,
                tool_choice="required",
                temperature=0
            )
            
            # 检查响应结构是否完整
            if not response.choices or len(response.choices) == 0:
                if not training_mode:
                    print(f"   ⚠️ Warning: No choices in response")
                return []
            
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                if not training_mode:
                    print(f"   ⚠️ Warning: No message in choice")
                return []
            
            tool_calls = choice.message.tool_calls
            if not tool_calls: return []

            # 🌟 辅助函数: 还原 Memory ID
            def resolve_memory_id(simple_id):
                real = uuid_mapping.get(str(simple_id))
                if not real and not training_mode:
                    print(f"   ⚠️ Warning: LLM hallucinated Memory ID '{simple_id}', ignoring.")
                return real
            
            # 🌟 辅助函数: 还原 Fact ID
            def resolve_fact_id(simple_id):
                real = fact_id_mapping.get(str(simple_id))
                if not real and not training_mode:
                    print(f"   ⚠️ Warning: LLM hallucinated Fact ID '{simple_id}', ignoring.")
                return real

            for tool_call in tool_calls:
                try:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if not training_mode:
                        print(f"   🤖 Raw Action: {func_name} | Args: {args}")
                    decision = {"action": "NOOP"}

                    if func_name == "create_memory":
                        decision.update({
                            "action": "ADD", 
                            "summary": args.get("content", ""), 
                            "facts_to_link": args.get("evidence_facts", []),
                            "user_id": user_id
                        })
                    
                    elif func_name == "update_memory":
                        if "target_memory_id" in args:
                            real_tid = resolve_memory_id(args["target_memory_id"])
                            if real_tid:
                                orig_created = temp_mem_storage.get(real_tid, {}).get('created_at', int(time.time()))
                                decision.update({
                                    "action": "UPDATE", 
                                    "target_id": real_tid, 
                                    "new_content": args.get("new_content", ""), 
                                    "facts_to_link": args.get("evidence_facts", []), 
                                    "orig_created": orig_created,
                                    "user_id": user_id
                                })

                    elif func_name == "delete_memory":
                        if "target_memory_id" in args:
                            real_tid = resolve_memory_id(args["target_memory_id"])
                            if real_tid:
                                orig_created = temp_mem_storage.get(real_tid, {}).get('created_at', int(time.time()))
                                decision.update({
                                    "action": "DELETE", 
                                    "target_id": real_tid, 
                                    "facts_to_link": args.get("evidence_facts", []), 
                                    "orig_created": orig_created,
                                    "user_id": user_id
                                })

                    elif func_name == "infer_memory":
                        if "source_memory_ids" in args:
                            source_simples = args["source_memory_ids"]
                            # 确保source_simples是列表
                            if not isinstance(source_simples, list):
                                source_simples = [source_simples]
                            real_source_ids = [resolve_memory_id(sid) for sid in source_simples if resolve_memory_id(sid)]
                            if real_source_ids:
                                decision.update({
                                    "action": "INFER", 
                                    "source_ids": real_source_ids, 
                                    "summary": args.get("inference_content", ""), 
                                    "facts_to_link": args.get("evidence_facts", []),
                                    "user_id": user_id
                                })
                    
                    elif func_name == "fact_add":
                        # 查找对应的fact_index
                        content = args.get("content", "")
                        details = args.get("details", [])
                        fact_index = -1
                        # 根据content和details查找对应的新事实
                        for idx, fact in enumerate(all_new_facts):
                            if fact['text'] == content and json.dumps(fact['details'], sort_keys=True) == json.dumps(details, sort_keys=True):
                                fact_index = idx
                                break
                        
                        decision.update({
                            "action": "FACT_ADD", 
                            "content": content,
                            "details": details,
                            "fact_index": fact_index,
                            "user_id": user_id
                        })
                    
                    elif func_name == "fact_trajectorize":
                        if "target_fact_id" in args:
                            real_fact_id = resolve_fact_id(args["target_fact_id"])
                            if real_fact_id:
                                new_content = args.get("new_content", "")
                                details = args.get("details", [])
                                fact_index = -1
                                # 根据new_content和details查找对应的新事实
                                for idx, fact in enumerate(all_new_facts):
                                    if fact['text'] == new_content and json.dumps(fact['details'], sort_keys=True) == json.dumps(details, sort_keys=True):
                                        fact_index = idx
                                        break
                                
                                decision.update({
                                    "action": "FACT_TRAJECTORIZE", 
                                    "target_fact_id": real_fact_id, 
                                    "new_content": new_content,
                                    "diff": args.get("diff", ""),
                                    "details": details,
                                    "fact_index": fact_index,
                                    "user_id": user_id
                                })

                    elif func_name == "no_operation":
                        decision.update({"reason": args.get("reason", "No reason provided"), "user_id": user_id})
                    
                    if decision["action"] != "NOOP" or "reason" in decision:
                        all_decisions.append(decision)
                except Exception as e:
                    if not training_mode:
                        print(f"   ⚠️ Error processing tool call: {e}")
                    continue

        except Exception as e:
            if not training_mode:
                print(f"   ⚠️ Decision Error: {e}")
        
        return all_decisions
        
    # --- Batch Processing for Training with GRPO Support ---
    def batch_process(self, batch_data: List[Dict], user_id: str = 'default', grpo_compatible: bool = True) -> List[Dict]:
        """
        Batch processing for memory management training with GRPO compatibility.
        
        Args:
            batch_data (List[Dict]): List of input data for batch processing.
            user_id (str, optional): User ID for memory operations. Defaults to 'default'.
            grpo_compatible (bool, optional): Whether to return GRPO-compatible format. Defaults to True.
            
        Returns:
            List[Dict]: List of results for each input in the batch.
        """
        results = []
        
        for data in batch_data:
            # Extract facts from input text
            extract_result = self.step_extract(data['text'], extract_mode='whole')
            
            # Retrieve relevant memories
            context_bundles = self.step_retrieve(extract_result, limit=3, user_id=user_id)
            
            # Make decisions (memory operations) in training mode
            decisions = self.step_decide(extract_result, context_bundles, user_id=user_id, training_mode=True)
            
            # Execute decisions
            self.step_execute(decisions, extract_result, user_id=user_id)
            
            if grpo_compatible:
                # Format result for GRPO training
                result = {
                    'input': data['text'],
                    'extract_result': extract_result,
                    'decisions': decisions,
                    # Add GRPO-specific fields
                    'memory_operations': [d['action'] for d in decisions if d['action'] != 'NOOP'],
                    'memory_contents': [d.get('summary', '') for d in decisions if d['action'] != 'NOOP'],
                    # Ensure we have the expected_operation if provided in data
                    'expected_operation': data.get('expected_operation', '')
                }
            else:
                # Standard format for non-GRPO training
                result = {
                    'input': data['text'],
                    'extract_result': extract_result,
                    'decisions': decisions
                }
            
            results.append(result)
        
        return results

    # ==========================================
    # Step 4: Execute (Modified for Fact Inheritance)
    # ==========================================
    def step_execute(self, decisions: List[Dict], extract_result: Dict, user_id: str = 'default'):
        ts = int(time.time())
        chunk_id = extract_result['chunk_id']
        all_new_facts = extract_result['new_facts']
        
        # 1. 保存原始 Chunk
        self.client.insert(self.chunk_col, [{"chunk_id": chunk_id, "text": extract_result["chunk_text"], "timestamp": ts, "embedding": get_embedding(extract_result["chunk_text"])}])

        # 2. 收集所有要链接的事实文本
        all_facts_to_link = set()
        for decision in decisions:
            action = decision.get("action")
            facts_to_link = decision.get('facts_to_link', [])
            for fact_text in facts_to_link:
                all_facts_to_link.add(fact_text)
        
        # 3. 创建fact_id_map，用于记录fact_index与生成的fact ID之间的关系
        fact_id_map = {}
        # 首先为每个新事实添加fact_index
        for idx, fact in enumerate(all_new_facts):
            fact['fact_index'] = idx
        
        # 3. 对所有要处理的事实进行最终去重
        # 收集所有新事实
        all_facts = []
        for fact in all_new_facts:
            # 只处理在all_facts_to_link中的事实
            if fact['text'] in all_facts_to_link:
                all_facts.append(fact)
        
        # 对所有事实进行去重
        unique_all_facts = []
        seen_fact_keys = set()
        for fact in all_facts:
            fact_key = f"{fact['text']}::{json.dumps(fact['details'], sort_keys=True)}"
            # 也考虑去掉"User"前缀的情况
            stripped_fact_key = f"{fact['text'].lower().replace('user ', '')}::{json.dumps(fact['details'], sort_keys=True)}"
            if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                seen_fact_keys.add(stripped_fact_key)
                unique_all_facts.append(fact)
        
        if len(unique_all_facts) < len(all_facts):
            print(f"   ✅ 最终去重 {len(all_facts) - len(unique_all_facts)} 个重复事实")
        
        # 更新all_facts_to_link为去重后的事实文本集合
        all_facts_to_link = {fact['text'] for fact in unique_all_facts}

        # 初始化操作次数计数器，添加新的操作类型
        self.operation_counts.setdefault("FACT_ADD", 0)
        self.operation_counts.setdefault("FACT_TRAJECTORIZE", 0)
        
        # 3. 处理每个决策
        has_non_noop_action = False
        
        # 收集所有要链接的事实，确保去重
        all_matched_facts = []
        seen_fact_keys = set()
        
        for decision in decisions:
            action = decision.get("action")
            if action == "NOOP":
                self.operation_counts["NOOP"] += 1
                print(f"   🚫 No operation: {decision.get('reason', 'No reason provided')}")
                continue

            has_non_noop_action = True
            target_mem_id = None
            relations = []

            # --- CASE 1: ADD MEMORY ---
            if action == "ADD":
                self.operation_counts["ADD"] += 1
                target_mem_id = str(uuid.uuid4())
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", [], decision.get('user_id', 'default'))
                print(f"   ✅ Created Mem: {target_mem_id[:8]}... | Content: {decision['summary']}")

            # --- CASE 2: UPDATE MEMORY ---
            elif action == "UPDATE":
                self.operation_counts["UPDATE"] += 1
                target_mem_id = decision['target_id']
                
                # 查询旧的memory内容
                old_memories = self.client.query(
                    collection_name=self.mem_col,
                    filter=f"memory_id == '{target_mem_id}'",
                    output_fields=["content", "created_at"]
                )
                
                old_content = "" if not old_memories else old_memories[0].get("content", "")
                new_content = decision['new_content']
                
                # 记录update前后的内容
                print(f"   🔄 Updating Mem: {target_mem_id[:8]}...")
                print(f"      Before: {old_content[:]}...")
                print(f"      After:  {new_content[:]}...")
                
                self._upsert_mem(target_mem_id, new_content, decision['orig_created'], ts, "active", [], decision.get('user_id', 'default'))

            # --- CASE 3: DELETE MEMORY ---
            elif action == "DELETE":
                self.operation_counts["DELETE"] += 1
                target_mem_id = decision['target_id']
                self._upsert_mem(target_mem_id, "(Archived)", decision['orig_created'], ts, "archived", [], decision.get('user_id', 'default'))
                print(f"   ❌ Deleted Mem: {target_mem_id[:8]}...")

            # --- CASE 4: INFER MEMORY (With Fact Inheritance) ---
            elif action == "INFER":
                self.operation_counts["INFER"] += 1
                target_mem_id = str(uuid.uuid4()) # 这是 Memory C
                source_ids = decision.get('source_ids', []) # 这是 [A, B]
                #############################################################
                # 查询source_ids对应的memory内容，用于打印
                source_mems = []
                if source_ids:
                    quoted_source_ids = [f'"{sid}"' for sid in source_ids]
                    mem_filter = f"status == 'active' and memory_id in [{','.join(quoted_source_ids)}]"
                    try:
                        source_mems = self.client.query(
                            collection_name=self.mem_col,
                            filter=mem_filter,
                            output_fields=["content", "memory_id", "created_at", "user_id"]
                        )
                    except Exception as e:
                        print(f"   ⚠️ 查询source memory失败: {e}")
                #############################################################
                # 4.1 创建新记忆 C，并记录血缘关系 (inferred_from)
                relations = [{"type": "inferred_from", "target_id": sid} for sid in source_ids]
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", relations, decision.get('user_id', 'default'))
                ####################################################################################
                # 将infer前后的memory内容拼在同一个字符串里输出
                infer_output = f"   💡 Inferred Mem: {target_mem_id[:8]}... | From: {[s[:8] for s in source_ids]}\n"
                infer_output += f"   ┌─────────────────────────────────────────────────────────────────────────────────\n"
                
                # 拼接infer前的memory内容
                if source_mems:
                    infer_output += f"   │ 📋 Infer 前的 Memory ({len(source_mems)}个):\n"
                    for mem in source_mems:
                        mem_id = mem.get("memory_id", "unknown")
                        content = mem.get("content", "")
                        infer_output += f"   │      📌 ID: {mem_id[:8]}... | 内容: {content[:]}...\n"
                
                # 拼接infer生成的memory内容
                infer_output += f"   │ 📝 Infer生成的 Memory:\n"
                infer_output += f"   │      📌 ID: {target_mem_id[:8]}... | 内容: {decision['summary'][:]}...\n"
                infer_output += f"   └─────────────────────────────────────────────────────────────────────────────────"
                
                # 一次性输出整个字符串
                print(infer_output)
                #################################################################################
                # 4.2 🌟 核心修改：继承旧 Facts
                # 逻辑：找出所有支持 A 或 B 的 Fact，把 C 也加到它们的支持列表里
                if source_ids:
                    # 构建查询表达式：array_contains(linked_memory_ids, 'A') or ...
                    expr_parts = [f'array_contains(linked_memory_ids, "{sid}")' for sid in source_ids]
                    filter_expr = " || ".join(expr_parts)
                    
                    try:
                        # 查出旧 Facts
                        old_related_facts = self.client.query(
                            collection_name=self.fact_col,
                            filter=filter_expr,
                            output_fields=["fact_id", "linked_memory_ids", "text", "linked_chunk_id", "timestamp", "details", "embedding"]
                        )
                        
                        if old_related_facts:
                            updated_rows = []
                            for fact in old_related_facts:
                                current_links = fact.get("linked_memory_ids", [])
                                # 如果 C 还没关联上，就加进去
                                if target_mem_id not in current_links:
                                    current_links.append(target_mem_id)
                                    # 确保facts包含details字段
                                    if "details" not in fact:
                                        fact["details"] = []
                                    # 确保facts包含embedding字段
                                    if "embedding" not in fact or not isinstance(fact["embedding"], list):
                                        text = fact.get("text", "")
                                        details = fact.get("details", [])
                                        fact["embedding"] = self._generate_fact_embedding(text, details)
                                    updated_rows.append(fact)
                            
                            # 写回数据库 (Upsert)
                            if updated_rows:
                                self.client.upsert(self.fact_col, updated_rows)
                                print(f"      ↳ 🧬 Inherited {len(updated_rows)} old facts from sources.")
                    except Exception as e:
                        print(f"      ⚠️ Error inheriting facts: {e}")
            
            # --- CASE 5: ADD FACT ---
            elif action == "FACT_ADD":
                self.operation_counts["FACT_ADD"] += 1
                content = decision['content']
                details = decision.get('details', [])
                fact_index = decision.get('fact_index', -1)  # 获取fact_index
                
                # 检查是否已有该fact_index对应的fact_id
                if fact_index >= 0 and fact_index in fact_id_map:
                    # 如果已有，使用已有的fact_id
                    fact_id = fact_id_map[fact_index]
                    print(f"   🔄 使用已有的 Fact ID: {fact_id[:8]}... | Content: {content}")
                else:
                    # 否则生成新的fact_id
                    fact_id = str(uuid.uuid4())
                    # 记录fact_index与fact_id的映射关系
                    if fact_index >= 0:
                        fact_id_map[fact_index] = fact_id
                
                # 保存新事实到数据库，包含details
                fact = {
                    "fact_id": fact_id,
                    "linked_memory_ids": [],  # 初始不关联任何记忆
                    "linked_chunk_id": chunk_id,
                    "text": content,
                    "details": details,  # 保存传入的details
                    "timestamp": ts,
                    "user_id": decision.get('user_id', user_id),
                    "embedding": self._generate_fact_embedding(content, details),
                    "trajectory": []  # 初始空轨迹
                }
                
                self.client.upsert(self.fact_col, [fact])
                print(f"   ✅ Added Fact: {fact_id[:8]}... | Content: {content}")
                if details:
                    print(f"      Details: {json.dumps(details, ensure_ascii=False, indent=2)}")
            
            # --- CASE 6: UPDATE FACT (Fact Trajectorize) ---
            elif action == "FACT_TRAJECTORIZE":
                self.operation_counts["FACT_TRAJECTORIZE"] += 1
                target_fact_id = decision['target_fact_id']
                new_content = decision['new_content']
                diff = decision['diff']
                details = decision.get('details', [])
                fact_index = decision.get('fact_index', -1)  # 获取fact_index
                
                # 检查是否已有该fact_index对应的fact_id
                if fact_index >= 0 and fact_index in fact_id_map:
                    # 如果已有，使用已有的fact_id
                    target_fact_id = fact_id_map[fact_index]
                    print(f"   🔄 使用已有的 Fact ID: {target_fact_id[:8]}... 进行trajectorize操作")
                
                # 查询旧的fact内容
                old_facts = self.client.query(
                    collection_name=self.fact_col,
                    filter=f"fact_id == '{target_fact_id}'",
                    output_fields=["fact_id", "linked_memory_ids", "linked_chunk_id", "text", "details", "timestamp", "user_id", "embedding", "trajectory"]
                )
                
                if not old_facts:
                    print(f"   ⚠️ Fact not found: {target_fact_id}")
                    continue
                
                old_fact = old_facts[0]
                old_content = old_fact.get("text", "")
                old_details = old_fact.get("details", [])
                
                # 构建更新后的fact
                updated_fact = old_fact.copy()
                updated_fact["text"] = new_content
                updated_fact["details"] = details  # 更新details
                updated_fact["timestamp"] = ts
                
                # 更新trajectory字段，添加新的变化记录，只包含diff
                trajectory = updated_fact.get("trajectory", [])
                trajectory.append({
                    "timestamp": ts,
                    "diff": diff
                })
                updated_fact["trajectory"] = trajectory
                
                # 更新embedding
                updated_fact["embedding"] = self._generate_fact_embedding(new_content, details)
                
                # 写回数据库
                self.client.upsert(self.fact_col, [updated_fact])
                print(f"   🔄 Trajectorized Fact: {target_fact_id[:8]}...")
                print(f"      Diff: {diff}")
                print(f"      Before: {old_content[:]}...")
                if old_details:
                    print(f"      Old Details: {json.dumps(old_details, ensure_ascii=False, indent=2)}")
                print(f"      After:  {new_content[:]}...")
                if details:
                    print(f"      New Details: {json.dumps(details, ensure_ascii=False, indent=2)}")
                print(f"      Trajectory Length: {len(trajectory)}")
                
                # 更新fact_id_map，确保其他操作使用更新后的fact_id
                if fact_index >= 0:
                    fact_id_map[fact_index] = target_fact_id

            # --- Common: Link NEW Facts for this decision ---
            # 无论是 ADD, UPDATE 还是 INFER，都会把当前决策的新证据关联上去
            facts_to_link = decision.get('facts_to_link', [])
            if target_mem_id and facts_to_link:
                # 查找与待链接事实文本匹配的完整事实对象（包含details和fact_id）
                for fact_text in facts_to_link:
                    # 在所有新事实中查找匹配的文本，以获取完整的fact对象（包含details和fact_id）
                    matching_fact = next((f for f in all_new_facts if f['text'] == fact_text), None)
                    if matching_fact:
                        # 检查事实是否已经被处理过
                        fact_key = f"{matching_fact['text']}::{json.dumps(matching_fact['details'], sort_keys=True)}"
                        # 也考虑去掉"User"前缀的情况
                        stripped_fact_key = f"{matching_fact['text'].lower().replace('user ', '')}::{json.dumps(matching_fact['details'], sort_keys=True)}"
                        if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                            seen_fact_keys.add(fact_key)
                            seen_fact_keys.add(stripped_fact_key)
                            # 添加目标记忆ID到事实中
                            fact_with_target = matching_fact.copy()
                            fact_with_target['target_mem_id'] = target_mem_id
                            all_matched_facts.append(fact_with_target)
                    else:
                        # 如果没有找到匹配的完整事实对象，使用文本创建一个简单的事实对象
                        new_fact = {'text': fact_text, 'details': [], 'fact_id': str(uuid.uuid4()), 'target_mem_id': target_mem_id}
                        fact_key = f"{new_fact['text']}::{json.dumps(new_fact['details'], sort_keys=True)}"
                        stripped_fact_key = f"{new_fact['text'].lower().replace('user ', '')}::{json.dumps(new_fact['details'], sort_keys=True)}"
                        if fact_key not in seen_fact_keys and stripped_fact_key not in seen_fact_keys:
                            seen_fact_keys.add(fact_key)
                            seen_fact_keys.add(stripped_fact_key)
                            all_matched_facts.append(new_fact)
        
        # 批量处理所有匹配的事实，确保每个事实只被关联到相应的记忆
        if all_matched_facts:
            # 按事实ID分组
            facts_by_id = {}
            for fact in all_matched_facts:
                fact_id = fact['fact_id']
                if fact_id not in facts_by_id:
                    facts_by_id[fact_id] = {
                        'fact': fact,
                        'target_mem_ids': set()
                    }
                facts_by_id[fact_id]['target_mem_ids'].add(fact['target_mem_id'])
            
            # 构建要写入数据库的行
            rows = []
            for fact_info in facts_by_id.values():
                fact = fact_info['fact']
                fact_id = fact['fact_id']  # 获取当前fact的fact_id
                target_mem_ids = list(fact_info['target_mem_ids'])
                
                rows.append({
                    "fact_id": fact_id,
                    "linked_memory_ids": target_mem_ids, # 关联到所有相关的记忆
                    "linked_chunk_id": chunk_id,
                    "text": fact['text'],
                    "details": fact['details'],  # 保存事实的details信息
                    "timestamp": ts,
                    "user_id": decision.get('user_id', user_id),
                    "embedding": self._generate_fact_embedding(fact['text'], fact['details']),
                    "trajectory": []  # 添加trajectory字段，记录变化轨迹
                })
            
            if rows:
                self.client.upsert(self.fact_col, rows)
                print(f"   🔗 批量关联 {len(rows)} 个事实到对应记忆")

        # 4. 处理未关联到任何记忆的新事实（当所有决策都是NOOP时）
        # 找出所有未被关联的新事实
        unlinked_facts = []
        for fact in all_new_facts:
            if fact['text'] not in all_facts_to_link:
                unlinked_facts.append(fact)

        # 如果有未关联的新事实，直接保存到fact_col集合中
        if unlinked_facts:
            rows = []
            for fact in unlinked_facts:
                # 使用预处理步骤中生成的fact_id，而不是重新生成
                fact_id = fact.get('fact_id', str(uuid.uuid4()))
                rows.append({
                    "fact_id": fact_id,
                    "linked_memory_ids": [],  # 不关联到任何记忆
                    "linked_chunk_id": chunk_id,
                    "text": fact['text'],
                    "details": fact['details'],  # 保存事实的details信息
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(fact['text'], fact['details']),
                    "trajectory": []  # 添加trajectory字段，记录变化轨迹
                })
            if rows:
                self.client.upsert(self.fact_col, rows)
                print(f"   💾 Saved {len(rows)} unlinked facts to database...")

        # 5. 处理所有决策都是NOOP但有新事实的情况
        # 如果所有决策都是NOOP，且有新事实，确保它们都被保存
        if not has_non_noop_action and all_new_facts:
            # 检查是否还有未保存的新事实
            all_fact_texts = {fact['text'] for fact in all_new_facts}
            saved_fact_texts = all_facts_to_link  # 已经通过决策关联的事实
            unsaved_fact_texts = all_fact_texts - saved_fact_texts
            
            if unsaved_fact_texts:
                rows = []
                for fact in all_new_facts:
                    if fact['text'] in unsaved_fact_texts:
                        # 使用预处理步骤中生成的fact_id，而不是重新生成
                        fact_id = fact.get('fact_id', str(uuid.uuid4()))
                        rows.append({
                            "fact_id": fact_id,
                            "linked_memory_ids": [],  # 不关联到任何记忆
                            "linked_chunk_id": chunk_id,
                            "text": fact['text'],
                            "details": fact['details'],  # 保存事实的details信息
                            "timestamp": ts,
                            "user_id": user_id,
                            "embedding": self._generate_fact_embedding(fact['text'], fact['details']),
                            "trajectory": []  # 添加trajectory字段，记录变化轨迹
                        })
                if rows:
                    self.client.upsert(self.fact_col, rows)
                    print(f"   💾 Saved {len(rows)} unlinked facts to database (all actions were NOOP)...")
                    
    def _upsert_mem(self, mem_id, content, c_at, u_at, status, relations, user_id):
        self.client.upsert(self.mem_col, [{
            "memory_id": mem_id,
            "embedding": get_embedding(content),
            "content": content,
            "user_id": user_id,
            "status": status,
            "created_at": c_at,
            "updated_at": u_at,
            "relations": relations
        }])

    def step_preprocess_facts(self, extract_result: Dict, user_id: str = 'default') -> Dict:
        """
        预处理提取出的事实，检查是否已存在于数据库中，确保从源头上去重
        
        Args:
            extract_result: 提取结果字典，包含new_facts
            user_id: 用户标识，确保只处理当前用户的事实
            
        Returns:
            更新后的提取结果字典，包含fact_id信息
        """
        new_facts = extract_result['new_facts']
        processed_facts = []
        ts = extract_result['timestamp']
        chunk_id = extract_result['chunk_id']
        
        print(f"🔍 [Preprocess Facts] 检查 {len(new_facts)} 个事实是否已存在...")
        
        # 1. 先对同一批次内的事实进行去重，避免同一批次中重复的事实被处理
        unique_facts_in_batch = []
        seen_fact_keys = set()
        for fact in new_facts:
            # 使用fact_text和details的组合作为唯一标识
            fact_key = f"{fact['text']}::{json.dumps(fact['details'], sort_keys=True)}"
            if fact_key not in seen_fact_keys:
                seen_fact_keys.add(fact_key)
                unique_facts_in_batch.append(fact)
        
        if len(unique_facts_in_batch) < len(new_facts):
            print(f"   ✅ 同一批次内去重 {len(new_facts) - len(unique_facts_in_batch)} 个重复事实")
        
        for fact in unique_facts_in_batch:
            fact_text = fact['text']
            fact_details = fact['details']

            
            # 3. 查询数据库中是否存在相同的fact
            existing_fact = None
            try:
                # 先尝试搜索相关事实，避免全量查询
                # 使用更安全的查询方式，基于text的前缀匹配
                # 只查询text字段包含fact_text关键词的事实
                search_vec = get_embedding(fact_text)
                search_results = self.client.search(
                    self.fact_col, [search_vec], 
                    output_fields=["fact_id", "details", "timestamp", "linked_memory_ids", "linked_chunk_id", "text"],
                    limit=20,  # 只查询前20个最相似的事实
                    similarity_threshold=0.8  # 设置相似度阈值，只返回相似度较高的事实
                )
                
                # 处理搜索结果，检查是否有完全匹配的事实
                if search_results and search_results[0]:
                    for hit in search_results[0]:
                        res = hit['entity']
                        res_text = res.get("text", "")
                        res_details = res.get("details", [])
                        # 检查是否是相同的事实，考虑到表述可能略有不同
                        # 1. 完全相同的情况
                        if res_text == fact_text and res_details == fact_details:
                            existing_fact = res
                            break
                        # 2. 核心内容相同但表述略有不同的情况（如有无"User"前缀）
                        stripped_res_text = res_text.lower().replace("user ", "").strip()
                        stripped_fact_text = fact_text.lower().replace("user ", "").strip()
                        if stripped_res_text == stripped_fact_text and res_details == fact_details:
                            existing_fact = res
                            break
            
            except Exception as e:
                print(f"   ⚠️ 查询事实时发生错误: {e}")
            
            if existing_fact:
                # 事实已存在，更新timestamp
                fact_id = existing_fact["fact_id"]
                old_ts = existing_fact["timestamp"]
                
                # 获取现有的linked_memory_ids和linked_chunk_id
                existing_links = existing_fact.get("linked_memory_ids", [])
                existing_chunk = existing_fact.get("linked_chunk_id", "")
                
                # 更新timestamp和关联信息
                self.client.upsert(self.fact_col, [{
                    "fact_id": fact_id,
                    "linked_memory_ids": existing_links,
                    "linked_chunk_id": existing_chunk,
                    "text": fact_text,
                    "details": fact_details,
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(fact_text, fact_details)
                }])
                
                # 将现有事实添加到processed_facts
                processed_fact = {
                    "text": fact_text,
                    "details": fact_details,
                    "fact_id": fact_id
                }
                processed_facts.append(processed_fact)
                
                print(f"   🔄 事实已存在，更新timestamp: {fact_id} (旧: {old_ts}, 新: {ts})")
            else:
                # 事实不存在，生成新的fact_id并保存
                fact_id = str(uuid.uuid4())
                # print(f"   🆕 新事实: {fact_id}")
                
                # 保存新事实到数据库
                self.client.upsert(self.fact_col, [{
                    "fact_id": fact_id,
                    "linked_memory_ids": [],
                    "linked_chunk_id": chunk_id,
                    "text": fact_text,
                    "details": fact_details,
                    "timestamp": ts,
                    "user_id": user_id,
                    "embedding": self._generate_fact_embedding(fact_text, fact_details)
                }])
                
                processed_fact = {
                    "text": fact_text,
                    "details": fact_details,
                    "fact_id": fact_id
                }
                
                processed_facts.append(processed_fact)
        
        # 更新提取结果
        extract_result['new_facts'] = processed_facts
        return extract_result
    
    def process(self, text, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None):
        res = self.step_extract(text, extract_mode=extract_mode)
        if not res['new_facts']: return
        
        # 预处理事实，检查是否已存在
        res = self.step_preprocess_facts(res, user_id=user_id)
        
        # 检查预处理后是否还有新事实
        if not res['new_facts']:
            print(f"   ✅ 所有事实都已存在，无需处理")
            return
        
        print(f"   新证据: {res['new_facts']}")
        
        # 检索相关记忆
        context_bundles = self.step_retrieve(res, limit=retrieve_limit, user_id=user_id, similarity_threshold=similarity_threshold)
        
        # 生成决策
        decisions = self.step_decide(res, context_bundles, user_id=user_id)
        
        # 执行决策
        self.step_execute(decisions, res, user_id=user_id)
        
        # 返回操作结果（可选）
        return decisions

    def _generate_fact_embedding(self, text: str, details: List) -> List[float]:
        """
        为事实生成嵌入向量，考虑text和details
        
        Args:
            text: 事实文本
            details: 事实详情列表
            
        Returns:
            嵌入向量列表
        """
        # 将text和details合并为一个字符串进行嵌入
        combined_text = text
        if details:
            # 将details转换为字符串，保留关键信息
            details_str = " ".join([str(detail) for detail in details])
            combined_text = f"{text} {details_str}"
        
        return get_embedding(combined_text)

    def get_operation_counts(self):
        """获取操作次数统计"""
        return self.operation_counts

    def reset_operation_counts(self):
        """重置操作次数统计"""
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}

    def get_collections(self):
        """获取集合名称"""
        return {
            "memories": self.mem_col,
            "facts": self.fact_col,
            "chunks": self.chunk_col
        }

    def drop_collections(self):
        """删除所有集合"""
        self.client.drop_collection(self.mem_col)
        self.client.drop_collection(self.fact_col)
        self.client.drop_collection(self.chunk_col)
        print("All collections dropped.")

    def get_facts_by_memory_id(self, memory_id: str) -> List[Dict]:
        """
        获取关联到指定记忆的所有事实
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            事实列表
        """
        try:
            res = self.client.query(
                collection_name=self.fact_col,
                filter=f'array_contains(linked_memory_ids, "{memory_id}")',
                output_fields=["fact_id", "text", "details", "timestamp", "linked_chunk_id"]
            )
            return res
        except Exception as e:
            print(f"Error getting facts by memory id: {e}")
            return []

    def get_memories_by_fact_id(self, fact_id: str) -> List[Dict]:
        """
        获取关联到指定事实的所有记忆
        
        Args:
            fact_id: 事实ID
            
        Returns:
            记忆列表
        """
        try:
            # 先获取事实的linked_memory_ids
            fact_res = self.client.query(
                collection_name=self.fact_col,
                filter=f'fact_id == "{fact_id}"',
                output_fields=["linked_memory_ids"]
            )
            
            if not fact_res:
                return []
            
            linked_memory_ids = fact_res[0].get("linked_memory_ids", [])
            if not linked_memory_ids:
                return []
            
            # 构建查询条件
            quoted_ids = [f'"{mem_id}"' for mem_id in linked_memory_ids]
            filter_expr = f"memory_id in [{','.join(quoted_ids)}] and status == 'active'"
            
            # 查询关联的记忆
            mem_res = self.client.query(
                collection_name=self.mem_col,
                filter=filter_expr,
                output_fields=["memory_id", "content", "created_at", "updated_at", "relations"]
            )
            
            return mem_res
        except Exception as e:
            print(f"Error getting memories by fact id: {e}")
            return []
    
    def search_memories(self, query: str, top_k: int = 20, user_id: str = 'default', threshold: float = 0.0, enhanced_search: bool = False) -> List[Dict]:
        """
        搜索与查询相关的记忆，并返回带有相关事实的结果
        
        Args:
            query: 查询文本
            top_k: 返回的记忆数量
            user_id: 用户ID，用于过滤记忆
            threshold: 相似度阈值
            enhanced_search: 是否启用增强型搜索
            
        Returns:
            带有相关事实的记忆列表
        """
        # 生成查询向量
        query_vec = get_embedding(query)
        
        # 搜索相关的memories
        mem_res = self.client.search(
            self.mem_col, [query_vec], filter=f"status == 'active' and user_id == '{user_id}'", limit=top_k,
            output_fields=["content", "memory_id", "created_at"],
            similarity_threshold=threshold
        )
        
        results = []
        if mem_res and mem_res[0]:
            for hit in mem_res[0]:
                mem = hit['entity']
                mem['similarity'] = hit['distance']
                
                # 获取相关的facts
                related_facts = self.get_facts_by_memory_id(mem['memory_id'])
                for fact in related_facts:
                    fact_vec = get_embedding(fact['text'])
                    # 计算事实与查询的相似度
                    similarity = np.dot(query_vec, fact_vec)
                    fact['similarity'] = similarity
                
                mem['related_facts'] = related_facts
                results.append(mem)
        
        return results
    
    def generate_response(self, question, question_date, context):
        """
        生成问题响应
        
        Args:
            question: 问题文本
            question_date: 问题日期
            context: 上下文信息
            
        Returns:
            LLM响应对象
        """
        prompt = LME_ANSWER_PROMPT.format(
            question=question,
            question_date=question_date,
            context=context
        )
        response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0,
                )
        
        return response
    
    def process_user_memory_infer(self, line, retrieve_limit=3, extract_mode="whole", user_id="default"):
        """
        处理用户记忆会话，包括提取、检索、决策和执行步骤
        
        Args:
            line: 包含用户记忆会话的字典
            retrieve_limit: 检索记忆的数量限制
            extract_mode: 提取模式
            user_id: 用户ID
            
        Returns:
            操作统计信息
        """
        # 获取context（即记忆会话文本）
        context = line.get("context", "")
        if not context:
            # 如果没有context，尝试从haystack_dates中获取（longmemeval数据集格式）
            haystack_dates = line.get("haystack_dates", [])
            if haystack_dates and isinstance(haystack_dates, list):
                # 合并所有haystack_dates中的文本
                context = "\n".join([item["text"] for item in haystack_dates if isinstance(item, dict) and "text" in item])
        
        # 如果还是没有context，跳过处理
        if not context:
            print(f"   ⚠️ 没有找到context，跳过处理")
            return {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        
        # 提取事实
        extract_result = self.step_extract(context, extract_mode=extract_mode)
        
        # 如果没有提取到事实，跳过处理
        if not extract_result.get("new_facts"):
            print(f"   ⚠️ 没有提取到事实，跳过处理")
            return {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        
        # 检索相关记忆和事实
        context_bundles = self.step_retrieve(extract_result, limit=retrieve_limit, user_id=user_id)
        
        # 生成决策
        decisions = self.step_decide(extract_result, context_bundles, user_id=user_id)
        
        # 执行决策
        self.step_execute(decisions, extract_result, user_id=user_id)
        
        # 返回操作统计信息
        return self.get_operation_counts()

# ==========================================
# 评估相关函数
# ==========================================
def response_user(line, pipeline, retrieve_limit=20, max_facts_per_memory=3, user_id='default', threshold: float = 0.0, enhanced_search: bool = False):
    """处理用户问题，生成响应
    
    Args:
        line: 包含问题和其他信息的字典
        pipeline: MemoryPipeline实例
        retrieve_limit: 检索记忆的数量限制
        max_facts_per_memory: 每个记忆的事实数量限制
        user_id: 用户标识，确保只检索当前用户的记忆
        threshold: 相似度阈值，低于该阈值的记忆将被过滤掉
        enhanced_search: 是否启用增强型搜索模式，启用后会调大topk并增强rerank
    """
    question = line.get("question")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    
    # 增强型搜索模式：调大topk
    if enhanced_search:
        # 调大初始检索数量，例如乘以2
        enhanced_top_k = retrieve_limit * 2
        print(f"   🚀 启用增强型搜索模式，初始检索数量: {enhanced_top_k}")
    else:
        enhanced_top_k = retrieve_limit
    
    # 搜索记忆，传递user_id、threshold和enhanced_search参数
    retrieved_memories = pipeline.search_memories(question, top_k=enhanced_top_k, user_id=user_id, threshold=threshold, enhanced_search=enhanced_search)
    
    # 确保retrieved_memories不是None
    retrieved_memories = retrieved_memories or []
    
    # 构建上下文，包含记忆和关联的事实
    memories_with_facts = []
    
    for mem in retrieved_memories:
        # 添加记忆内容
        memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
        memories_with_facts.append(memory_line)
        
        # 添加关联的事实（如果有）
        related_facts = mem.get("related_facts", [])
        if related_facts:
            # 直接使用search_memories中已经计算好的相似度分数
            fact_with_scores = []
            for fact in related_facts:
                # 获取已计算的相似度分数，默认为0
                similarity = fact.get("similarity", 0)
                fact_with_scores.append((fact, similarity))
            
            # 根据相关性分数对事实进行排序
            fact_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 添加排序后的事实，限制数量
            for i, (fact, score) in enumerate(fact_with_scores[:max_facts_per_memory]):
                # 优化事实输出格式
                fact_text = fact['text']
                details = fact['details']
                
                # 格式化细节
                if details:
                    # 将细节列表转换为更易读的格式
                    details_str = "; ".join(details)
                    # 如果细节太长，截断
                    if len(details_str) > 100:
                        details_str = details_str[:97] + "..."
                    fact_line = f"  ├── [{i+1}] 事实: {fact_text}\n  │     细节: {details_str}"
                else:
                    fact_line = f"  ├── [{i+1}] 事实: {fact_text}"
                
                memories_with_facts.append(fact_line)
    
    memories_str = "\n".join(memories_with_facts)
    
    # 生成响应
    response = pipeline.generate_response(question, question_date_string, memories_str)
    answer = response.choices[0].message.content
    
    return retrieved_memories, answer


def process_and_evaluate_user(line, user_index, infer=True, retrieve_limit: int = 3, extract_mode: str = "whole", vector_db_type="milvus", dataset_name=""):
    """
    封装单个用户的所有处理步骤，以便并行执行。
    返回一个包含所有统计信息的字典。
    """
    try:
        # 为每个用户生成唯一的user_id，确保记忆隔离
        user_id = f"user_{user_index}"
        
        # 为每个用户创建独立的pipeline实例，避免多线程竞争
        # 注意：每个用户的pipeline实例不应该清空数据库，clear_db固定为False
        pipeline = MemoryPipeline(vector_db_type=vector_db_type, clear_db=False, dataset_name=dataset_name)
        
        # 处理用户记忆会话，传递user_id
        memory_counts = pipeline.process_user_memory_infer(line, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id)
        
        # 生成问题响应，传递user_id
        retrieved_memories, answer = response_user(line, pipeline, retrieve_limit, user_id=user_id)
        
        # 确保retrieved_memories不是None
        retrieved_memories = retrieved_memories or []
        
        # 构建上下文字符串用于后续处理
        memories_with_facts = []
        
        # 生成查询向量，用于计算事实与查询的相关性
        query_vec = get_embedding(line.get("question", ""))
        
        for mem in retrieved_memories:
            # 添加记忆内容
            memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
            memories_with_facts.append(memory_line)

            # 添加关联的事实（如果有）
            related_facts = mem.get("related_facts", [])
            max_facts_per_memory = 3  # 每个记忆的事实数量限制
            if related_facts:
                # 计算每个事实与查询的相关性分数
                fact_with_scores = []
                for fact in related_facts:
                    try:
                        fact_vec = get_embedding(fact["text"])
                        # 使用向量点积作为相关性分数
                        dot_product = sum(a * b for a, b in zip(query_vec, fact_vec))
                        fact_with_scores.append((fact, dot_product))
                    except Exception as e:
                        print(f"计算事实相关性失败: {e}")
                        fact_with_scores.append((fact, 0))
                
                # 根据相关性分数对事实进行排序
                # fact_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 添加排序后的事实，限制数量
                for i, (fact, score) in enumerate(fact_with_scores[:max_facts_per_memory]):
                    # 优化事实输出格式
                    fact_text = fact['text']
                    details = fact['details']
                    
                    # 格式化细节
                    if details:
                        # 将细节列表转换为更易读的格式
                        details_str = "; ".join(details)
                        # 如果细节太长，截断
                        if len(details_str) > 100:
                            details_str = details_str[:97] + "..."
                        fact_line = f"  ├── [{i+1}] 事实: {fact_text}\n  │     细节: {details_str}"
                    else:
                        fact_line = f"  ├── [{i+1}] 事实: {fact_text}"
                    
                    memories_with_facts.append(fact_line)
        
        memories_str = "\n".join(memories_with_facts)
        
        # 获取标准答案和问题类型
        golden_answer = line.get("answer")
        question = line.get("question")
        question_type = line.get("question_type", "unknown")
        
        # 评估答案正确性
        is_correct = lme_grader(llm_client, question, golden_answer, answer)
        
        return {
            "index": user_index,
            "is_correct": is_correct,
            "counts": memory_counts,
            "question": question,
            "question_type": question_type,
            "answer": answer,
            "golden_answer": golden_answer,
            "retrieved_memories": retrieved_memories,
            "context": memories_str,
        }
    except Exception as e:
        print(f"处理用户 {user_index} 出错 ({line.get('question', 'Unknown')[:20]}...): {e}")
        return {
            "index": user_index,
            "is_correct": False,
            "counts": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0},
            "question": line.get("question", "N/A"),
            "question_type": line.get("question_type", "unknown"),
            "context": "N/A",
            "answer": "N/A",
            "golden_answer": line.get("answer", "N/A"),
            "retrieved_memories": []
        }


# ==========================================
# Main Test & Evaluation
# ==========================================
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Memory Pipeline with longmemeval Evaluation")
    parser.add_argument("--eval", action="store_true", help="是否进行评估")
    parser.add_argument("--infer", action="store_true", default=True, help="是否使用推理功能")
    parser.add_argument("--num_users", type=int, default=50, help="评估用户数量")
    parser.add_argument("--max_workers", type=int, default=10, help="并行处理的工作线程数")
    parser.add_argument("--retrieve_limit", type=int, default=3, help="检索时返回的记忆数量")
    parser.add_argument("--threshold", type=float, default=0.7, help="记忆相似度阈值，低于该阈值的记忆将被过滤掉")
    parser.add_argument("--extract-mode", type=str, default="whole", choices=["whole", "turn"], help="提取模式：whole-对整个chunk进行提取，turn-按轮次提取")
    parser.add_argument("--vector-db-type", type=str, default="milvus", choices=["milvus", "qdrant"], help="指定使用的向量数据库类型")
    parser.add_argument("--clear-db", action="store_true", help="运行前清空数据库")
    parser.add_argument("--data-path", type=str, help="指定数据文件路径")
    parser.add_argument("--dataset-type", type=str, default="longmemeval", choices=["longmemeval", "hotpotqa"], help="指定数据集类型")
    args = parser.parse_args()
    
    # 初始化内存管道
    pipeline = MemoryPipeline(vector_db_type=args.vector_db_type, clear_db=args.clear_db, dataset_name=args.dataset_type)
    
    if args.eval:
        # 评估模式
        try:
            # 根据数据集类型设置默认数据路径
            if args.dataset_type == "hotpotqa":
                data_path = args.data_path or "./data/hotpotqa-val.jsonl"
            else:  # longmemeval
                data_path = args.data_path or "./data/longmemeval_s_cleaned.json"
                
            print(f"调试信息：")
            print(f"  数据集类型：{args.dataset_type}")
            print(f"  指定的数据路径：{args.data_path}")
            print(f"  实际使用的数据路径：{data_path}")
            print(f"  文件是否存在：{os.path.exists(data_path)}")
            
            if not os.path.exists(data_path):
                print(f"数据集文件不存在: {data_path}")
                exit()
            
            # 判断文件类型并加载数据
            lines = []
            print(f"  文件格式：{'JSONL' if data_path.endswith('.jsonl') else 'JSON'}")
            if data_path.endswith(".jsonl"):
                # 处理JSONL格式文件
                print(f"  开始加载JSONL文件...")
                with open(data_path, "r") as f:
                    for i, line in enumerate(f):
                        lines.append(json.loads(line.strip()))
                        if i < 2:  # 打印前2条数据的关键字段
                            loaded_item = lines[-1]
                            print(f"    第{i+1}条数据关键字段：")
                            print(f"      是否包含context：{'context' in loaded_item}")
                            print(f"      是否包含haystack_dates：{'haystack_dates' in loaded_item}")
                            print(f"      数据ID：{loaded_item.get('id', '未知')}")
            else:
                # 处理JSON格式文件
                print(f"  开始加载JSON文件...")
                with open(data_path, "r") as f:
                    data = json.load(f)
                    lines = data.get("items", []) if isinstance(data, dict) else data
                    
                    if len(lines) > 0:
                        # 打印前2条数据的关键字段
                        for i in range(min(2, len(lines))):
                            loaded_item = lines[i]
                            print(f"    第{i+1}条数据关键字段：")
                            print(f"      是否包含context：{'context' in loaded_item}")
                            print(f"      是否包含haystack_dates：{'haystack_dates' in loaded_item}")
                            print(f"      数据ID：{loaded_item.get('id', '未知')}")
            
            # 限制处理的用户数量
            lines = lines[:args.num_users] if args.num_users > 0 else lines
            print(f"  加载完成，共 {len(lines)} 条数据，准备处理...")
            
            # 初始化统计数据
            all_results = []
            total = len(lines)
            correct = 0
            
            # 并行处理用户数据
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 为每个用户提交一个任务
                future_to_index = {
                    executor.submit(process_and_evaluate_user, line, i, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type): i
                    for i, line in enumerate(lines)
                }
                
                # 处理完成的任务
                for future in tqdm(as_completed(future_to_index), total=total, desc="处理用户数据"):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        if result["is_correct"]:
                            correct += 1
                            print(f"\n✅ 用户 {index} 回答正确")
                        else:
                            print(f"\n❌ 用户 {index} 回答错误")
                            print(f"   问题: {result['question']}")
                            print(f"   预测: {result['answer']}")
                            print(f"   标准答案: {result['golden_answer']}")
                    except Exception as e:
                        print(f"\n⚠️ 用户 {index} 处理失败: {e}")
            
            # 计算准确率
            accuracy = correct / total if total > 0 else 0
            print(f"\n📊 评估结果：")
            print(f"   总用户数: {total}")
            print(f"   正确数: {correct}")
            print(f"   准确率: {accuracy:.4f}")
            
            # 保存评估结果
            result_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, "w") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"   评估结果已保存到: {result_file}")
            
        except Exception as e:
            print(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 测试模式，用于单条数据测试
        print("🚀 进入测试模式")
        
        # 示例测试数据
        test_data = {
            "context": "User bought a new laptop. It's a MacBook Pro with M3 chip.",
            "question": "What did the user buy?",
            "answer": "A new MacBook Pro with M3 chip."
        }
        
        # 处理测试数据
        print(f"\n📝 测试数据：")
        print(f"   Context: {test_data['context']}")
        print(f"   Question: {test_data['question']}")
        print(f"   Answer: {test_data['answer']}")
        
        # 提取事实
        extract_result = pipeline.step_extract(test_data['context'], extract_mode=args.extract_mode)
        print(f"\n🔍 提取结果：")
        print(f"   {json.dumps(extract_result, ensure_ascii=False, indent=2)}")
        
        # 检索相关记忆和事实
        context_bundles = pipeline.step_retrieve(extract_result, limit=args.retrieve_limit)
        print(f"\n🧠 检索结果：")
        for i, bundle in enumerate(context_bundles):
            print(f"   事实 {i+1}: {bundle['new_fact']['text']}")
            print(f"   相关记忆: {len(bundle['candidates'])}")
            print(f"   相关事实: {len(bundle['related_facts'])}")
        
        # 生成决策
        decisions = pipeline.step_decide(extract_result, context_bundles)
        print(f"\n📋 决策结果：")
        for decision in decisions:
            print(f"   {decision}")
        
        # 执行决策
        pipeline.step_execute(decisions, extract_result)
        print(f"\n✅ 执行完成")
        
        # 生成响应
        retrieved_memories, answer = response_user(test_data, pipeline, args.retrieve_limit)
        print(f"\n💬 生成回答：")
        print(f"   问题: {test_data['question']}")
        print(f"   回答: {answer}")
        print(f"   标准答案: {test_data['answer']}")
        
    # 打印操作统计信息
    print(f"\n📊 操作统计：")
    print(f"   {pipeline.get_operation_counts()}")
