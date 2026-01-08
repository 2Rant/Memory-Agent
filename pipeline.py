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

[MANDATORY OUTPUT FORMAT]
For every new fact you process, you MUST:
1. First generate a detailed thinking process
2. Then call the appropriate tool

[THINKING PROCESS REQUIREMENTS]
Your thinking process MUST include:
- The specific new fact you're analyzing
- Which existing memories are relevant (with their IDs)
- How memories are connected through related facts
- Your comparison and reasoning
- Which operation you've decided to perform and why

[OPERATIONS & GUIDELINES]
Compare New Facts with Existing Memories and perform the following operations using the available tools. 
DO NOT output raw JSON text. You MUST use the provided function tools.

1. **ADD (create_memory)**
   - **Condition**: If a fact contains completely NEW information not present in Existing Memories.
   - **Action**: Call `create_memory` with a concise summary of the facts, not just a simple concatenation.
   - **Important**: Memory content should be a meaningful and concise summary.
  
2. **UPDATE (update_memory)**
   - **Condition**: If a fact adds detail, corrects, or updates a specific Existing Memory.
   - **Constraint**: You MUST use the Integer ID (e.g., "0") provided in the input as the `target_memory_id`.
   - **Logic**: Merge the old content and new fact into a comprehensive statement, not just a simple concatenation.
   - **Example**:
     - Old (ID="0"): "User likes generic pizza."
     - New Fact: "User loves pepperoni pizza."
     - Action: `update_memory(target_memory_id="0", new_content="User loves pepperoni pizza", ...)`

3. **DELETE (delete_memory)**
   - **Condition**: If a fact explicitly contradicts an Existing Memory (and the new fact is trusted), or if the memory is no longer valid.
   - **Constraint**: Use the Integer ID (e.g., "1") as `target_memory_id`.

4. **INFER (infer_memory) [CRITICAL]**
   - **Condition**: Look for higher-level insights. If combining "Memory A" and "Memory B" reveals a hidden connection or causality.
   - **Action**: Call `infer_memory`.
   - **Example**:
     - Memory A (ID="2"): "User moved to Singapore."
     - Memory B (ID="3"): "User bought a Type G power adapter."
     - Inference: "User is preparing electronics for Singapore power standards."
     - Action: `infer_memory(source_memory_ids=["2", "3"], inference_content="...")`

5. **NOOP (no_operation)**
   - **Condition**: If the fact is redundant (already exactly covered by memory), similar to existing facts associated with the retrieved memories, or trivial.

[STRICT ID RULES]
- When calling `update_memory` or `delete_memory`, **ONLY** use the string integer IDs (e.g., "0", "1", "2") found in the [EXISTING MEMORIES] list.
- **NEVER** invent a UUID or use an ID that is not in the provided list.
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
            "name": "no_operation",
            "description": "No action needed if the fact is redundant (already exactly covered by memory or its associated facts).",
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
        """初始化MemoryPipeline
        
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
        
        # 检查并创建集合
        
        # 处理 memories 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.mem_col):
                # 创建完整的schema
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
                print(f"Collection '{self.mem_col}' created.")
                
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
                print(f"Collection '{self.mem_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.mem_col)
            print(f"Collection '{self.mem_col}' created or exists.")
        
        # 处理 facts 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.fact_col):
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("fact_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("linked_memory_ids", self.client.DataType.JSON)
                s.add_field("linked_chunk_id", self.client.DataType.VARCHAR, max_length=64)
                s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("details", self.client.DataType.JSON)  # 添加details字段
                s.add_field("timestamp", self.client.DataType.INT64)
                s.add_field("user_id", self.client.DataType.VARCHAR, max_length=64)  # 添加user_id字段
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                
                # 创建集合
                self.client.create_collection(self.fact_col, schema=s)
                print(f"Collection '{self.fact_col}' created.")
                
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
                print(f"Collection '{self.fact_col}' already exists, skipping creation.")
        else:
            # 非Milvus客户端，直接创建集合
            self.client.create_collection(self.fact_col)
            print(f"Collection '{self.fact_col}' created or exists.")
        
        # 处理 chunks 集合
        if hasattr(self.client, 'DataType'):
            # 这是 Milvus 客户端
            # 检查集合是否存在
            if not self.client.has_collection(self.chunk_col):
                s = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
                s.add_field("chunk_id", self.client.DataType.VARCHAR, max_length=64, is_primary=True)
                s.add_field("text", self.client.DataType.VARCHAR, max_length=65535)
                s.add_field("timestamp", self.client.DataType.INT64)
                s.add_field("embedding", self.client.DataType.FLOAT_VECTOR, dim=dim)
                
                # 创建集合
                self.client.create_collection(self.chunk_col, schema=s)
                print(f"Collection '{self.chunk_col}' created.")
                
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
                print(f"Collection '{self.chunk_col}' already exists, skipping creation.")
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
    def step_extract(self, chunk_text: str, extract_mode: str = "whole", timestamp: int = None) -> Dict:
        """
        从对话中提取事实
        
        Args:
            chunk_text: 对话文本
            extract_mode: 提取模式，可选值：
                - "whole": 对整个chunk进行提取
                - "turn": 按轮次提取，每轮user-assistant对话单独提取
            timestamp: 时间戳，可选，默认使用当前时间
        
        Returns:
            包含提取事实的字典
        """
        # print(f"\n👀 [1. Extract] Processing: '{chunk_text}'")
        
        # 如果没有提供timestamp，使用当前时间
        if timestamp is None:
            timestamp = int(time.time())
        
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
                    
                    return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": all_facts, "timestamp": timestamp}
            except Exception as e:
                print(f"解析对话轮次失败，回退到whole模式: {e}")
        
        # 默认模式：对整个chunk进行提取
        facts = self._extract_single_turn(chunk_text)
        return {"chunk_id": str(uuid.uuid4()), "chunk_text": chunk_text, "new_facts": facts, "timestamp": timestamp}
    
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
        
        print(f"🔍 [2. Retrieve] Searching Memories for {len(new_facts)} facts...")
        context_bundles = []

        for fact in new_facts:
            query_vec = get_embedding(fact['text'])
            # 添加user_id过滤，确保只检索当前用户的记忆
            res = self.client.search(
                self.mem_col, [query_vec], filter=f"status == 'active' and user_id == '{user_id}'", limit=limit,
                output_fields=["content", "memory_id", "created_at"],
                similarity_threshold=similarity_threshold
            )
            candidates = []
            if res and res[0]:
                for hit in res[0]:
                    candidates.append(hit['entity'])
            
            # 检索这些记忆关联的事实
            related_facts = []
            if candidates:
                # 获取所有候选记忆的ID
                memory_ids = [mem['memory_id'] for mem in candidates]
                # 构建查询条件，查找关联到这些记忆的事实
                expr_parts = [f'array_contains(linked_memory_ids, "{mem_id}")' for mem_id in memory_ids]
                filter_expr = " || ".join(expr_parts)
                
                try:
                    related_facts = self.client.query(
                        collection_name=self.fact_col,
                        filter=filter_expr,
                        output_fields=["fact_id", "linked_memory_ids", "text", "linked_chunk_id", "timestamp", "details"]
                    )
                except Exception as e:
                    print(f"   ⚠️ Error retrieving related facts: {e}")
            
            # 将related_facts添加到每个memory对象中
            for mem in candidates:
                mem_id = mem['memory_id']
                mem['related_facts'] = [f for f in related_facts if mem_id in f.get('linked_memory_ids', [])]
            
            context_bundles.append({
                "new_fact": fact,
                "candidates": candidates
            })
            
        return context_bundles

    # --- Step 3: Decide (With ID Mapping) ---
    def step_decide(self, extract_result: Dict, context_bundles: List[Dict], user_id: str = 'default', training_mode: bool = False) -> List[Dict]:
        all_new_facts = extract_result['new_facts']
        
        # 1. 合并去重 Candidates
        temp_mem_storage = {}
        for bundle in context_bundles:
            for mem in bundle['candidates']:
                temp_mem_storage[mem['memory_id']] = mem
        
        unique_memories_list = list(temp_mem_storage.values())
        if not training_mode:
            print(f"🧠 [3. Manager] Global Decide: {len(all_new_facts)} facts vs {len(unique_memories_list)} memories.")

        # 🌟 2. 构造 ID 映射 (Mapping Logic)
        uuid_mapping = {}  # { "0": "real-uuid", "1": "real-uuid" }
        candidates_str = ""

        if not unique_memories_list:
            candidates_str = "(No relevant memories found. Treat as new topic.)"
        else:
            for idx, mem in enumerate(unique_memories_list):
                simple_id = str(idx)
                real_uuid = mem['memory_id']
                uuid_mapping[simple_id] = real_uuid
                candidates_str += f"[Memory Item ID: {simple_id}]\n- Content: {mem['content']}\n"
                
                # 添加关联的facts
                related_facts = mem.get('related_facts', [])
                if related_facts:
                    candidates_str += "- Related Facts:\n"
                    for fact_idx, fact in enumerate(related_facts):
                        candidates_str += f"  - Fact {fact_idx + 1}: {fact['text']}\n"
                        # 添加fact的details
                        details = fact.get('details', [])
                        if details:
                            if isinstance(details, list):
                                for detail in details:
                                    if isinstance(detail, dict):
                                        detail_str = ", ".join([f"{k}: {v}" for k, v in detail.items()])
                                        candidates_str += f"    Detail: {detail_str}\n"
                                    else:
                                        candidates_str += f"    Detail: {detail}\n"
                            elif isinstance(details, dict):
                                detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
                                candidates_str += f"    Detail: {detail_str}\n"
                candidates_str += "\n"

        # 构造最终 Prompt
        system_msg = MEMORY_MANAGER_PROMPT

        # 只提取事实的text字段，不包含details，避免LLM将details当作独立事实
        fact_texts = [fact['text'] for fact in all_new_facts]
        
        user_content = f"""
        [New Facts Stream]
        {json.dumps(fact_texts, ensure_ascii=False)}
        
        [EXISTING MEMORIES]
        {candidates_str}
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

            # 🌟 辅助函数: 还原 ID
            def resolve_id(simple_id):
                real = uuid_mapping.get(str(simple_id))
                if not real and not training_mode:
                    print(f"   ⚠️ Warning: LLM hallucinated ID '{simple_id}', ignoring.")
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
                            real_tid = resolve_id(args["target_memory_id"])
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
                            real_tid = resolve_id(args["target_memory_id"])
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
                            real_source_ids = [resolve_id(sid) for sid in source_simples if resolve_id(sid)]
                            if real_source_ids:
                                decision.update({
                                    "action": "INFER", 
                                    "source_ids": real_source_ids, 
                                    "summary": args.get("inference_content", ""), 
                                    "facts_to_link": args.get("evidence_facts", []),
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
        # 使用extract_result中的timestamp，而不是当前时间
        ts = extract_result['timestamp']
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

            # --- CASE 1: ADD ---
            if action == "ADD":
                self.operation_counts["ADD"] += 1
                target_mem_id = str(uuid.uuid4())
                self._upsert_mem(target_mem_id, decision['summary'], ts, ts, "active", [], decision.get('user_id', 'default'))
                print(f"   ✅ Created Mem: {target_mem_id[:8]}... | Content: {decision['summary']}")

            # --- CASE 2: UPDATE ---
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

            # --- CASE 3: DELETE ---
            elif action == "DELETE":
                self.operation_counts["DELETE"] += 1
                target_mem_id = decision['target_id']
                self._upsert_mem(target_mem_id, "(Archived)", decision['orig_created'], ts, "archived", [], decision.get('user_id', 'default'))
                print(f"   ❌ Deleted Mem: {target_mem_id[:8]}...")

            # --- CASE 4: INFER (With Fact Inheritance) ---
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
                                    # 确保facts包含user_id字段
                                    if "user_id" not in fact:
                                        fact["user_id"] = user_id
                                    updated_rows.append(fact)
                            
                            # 写回数据库 (Upsert)
                            if updated_rows:
                                self.client.upsert(self.fact_col, updated_rows)
                                print(f"      ↳ 🧬 Inherited {len(updated_rows)} old facts from sources.")
                    except Exception as e:
                        print(f"      ⚠️ Error inheriting facts: {e}")

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
                    "embedding": self._generate_fact_embedding(fact['text'], fact['details'])
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
                    "embedding": self._generate_fact_embedding(fact['text'], fact['details'])
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
                            "embedding": self._generate_fact_embedding(fact['text'], fact['details'])
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
    
    def process(self, text, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None, timestamp: int = None):
        res = self.step_extract(text, extract_mode=extract_mode, timestamp=timestamp)
        if not res['new_facts']: return
        
        # 预处理事实，检查是否已存在
        res = self.step_preprocess_facts(res, user_id=user_id)
        
        # 检查预处理后是否还有新事实
        if not res['new_facts']:
            print(f"   ✅ 所有事实都已存在，无需处理")
            return
        
        print(f"   新证据: {res['new_facts']}")
        
        ctx_bundles = self.step_retrieve(res, limit=retrieve_limit, user_id=user_id, similarity_threshold=similarity_threshold)
        decisions = self.step_decide(res, ctx_bundles, user_id=user_id)
        self.step_execute(decisions, res, user_id=user_id)
        
    def process_user_memory_infer(self, line, retrieve_limit: int = 3, extract_mode: str = "whole", user_id: str = 'default', similarity_threshold: float = None):
        """处理用户记忆会话，支持longmemeval数据集格式"""
        # 重置操作计数，确保每个用户的计数独立
        self.operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
        dates = line.get("haystack_dates")
        sessions = line.get("haystack_sessions")

        for session_id, session in enumerate(sessions):
            date = dates[session_id] + " UTC"
            date_format = "%Y/%m/%d (%a) %H:%M UTC"
            date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
            # 生成timestamp
            timestamp = int(date_string.timestamp())
            
            parsed_messages = parse_messages(session)
            print(f"处理会话 {session_id + 1}/{len(sessions)}: {dates[session_id]}")
            
            # 使用现有的process方法处理会话消息，传递user_id、similarity_threshold和timestamp
            self.process(parsed_messages, retrieve_limit=retrieve_limit, extract_mode=extract_mode, user_id=user_id, similarity_threshold=similarity_threshold, timestamp=timestamp)
        
        # 返回操作次数统计
        return self.operation_counts
        
    def search_memories(self, query_text, top_k=5, fact_top_k=5, user_id: str = 'default', threshold: float = 0.0, similarity_threshold: float = None, enhanced_search: bool = False, use_fact_retrieval: bool = True):
        """搜索记忆并返回每个记忆关联的topk个事实，并根据关联事实进行rerank
        
        Args:
            query_text: 查询文本
            top_k: 返回的记忆数量上限
            fact_top_k: 每个记忆关联的事实数量上限
            user_id: 用户标识，确保只检索当前用户的记忆
            threshold: 相似度阈值，低于该阈值的记忆将被过滤掉
            similarity_threshold: 向量数据库搜索时的相似度阈值，低于该阈值的记忆将被过滤掉
            enhanced_search: 是否启用增强型搜索模式，启用后会增强rerank逻辑
            use_fact_retrieval: 是否使用事实检索模式，启用后会搜索事实集合并根据关联的memory_id获取更多记忆
        """
        query_vec = get_embedding(query_text)
        
        # 添加调试信息
        filter_expr = f"status == 'active' and user_id == '{user_id}'"
        print(f"   🔍 搜索过滤条件: {filter_expr}, 阈值: {threshold}, 向量搜索阈值: {similarity_threshold}")
        
        # ===========================
        # 1. 搜索记忆集合，获取memoryA
        # ===========================
        mem_res = self.client.search(
            self.mem_col, [query_vec], filter=filter_expr, limit=top_k,  # 搜索更多记忆，避免遗漏
            output_fields=["content", "memory_id", "created_at", "user_id"],  # 包含user_id字段用于调试
            similarity_threshold=similarity_threshold
        )
        
        # ===========================
        # 2. 搜索事实集合，获取相关事实
        # ===========================
        combined_items = []  # 存储memory和fact及其分数，用于统一排序
        memory_dict = {}  # 临时存储memory对象
        fact_dict = {}  # 临时存储fact对象
        
        # 先处理memoryA
        if mem_res and mem_res[0]:
            for hit in mem_res[0]:
                memory = hit['entity']
                memory_id = memory['memory_id']
                # 保存原始分数（内积）
                memory["original_score"] = hit['distance']
                memory_dict[memory_id] = memory
                # 将memory添加到combined_items中，用于统一排序
                combined_items.append({
                    "type": "memory",
                    "item": memory,
                    "score": hit['distance'],  # 使用内积作为分数
                    "memory_id": memory_id
                })
        
        # 处理fact
        if use_fact_retrieval:
            # 搜索事实集合
            fact_res = self.client.search(
                self.fact_col, [query_vec], filter=f"user_id == '{user_id}'", limit=top_k,  # 搜索更多事实，避免遗漏
                output_fields=["text", "timestamp", "fact_id", "details", "linked_memory_ids", "user_id", "embedding"]  # 添加embedding字段
            )
            
            if fact_res and fact_res[0]:
                for hit in fact_res[0]:
                    fact = hit['entity']
                    fact_id = fact['fact_id']
                    # 计算fact与query的内积
                    try:
                        # 直接使用数据库中存储的embedding，而不是重新计算
                        fact_vec = fact.get("embedding")
                        if not fact_vec or not isinstance(fact_vec, list):
                            # 如果没有embedding字段或不是列表，重新计算，使用text和details拼接
                            fact_vec = self._generate_fact_embedding(fact["text"], fact.get("details", []))
                    
                        fact_dot_product = sum(a * b for a, b in zip(query_vec, fact_vec))
                        fact["similarity"] = fact_dot_product
                        fact_dict[fact_id] = fact
                        
                        # 将fact添加到combined_items中，用于统一排序
                        combined_items.append({
                            "type": "fact",
                            "item": fact,
                            "score": fact_dot_product,  # 使用内积作为分数
                            "fact_id": fact_id,
                            "linked_memory_ids": fact.get("linked_memory_ids", [])
                        })
                    except Exception as e:
                        print(f"计算事实相关性失败: {e}")
                        continue
        
        # ===========================
        # 3. 统一排序所有item，取topk
        # ===========================
        # 按分数降序排序
        combined_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 取topk*2个item，确保包含足够多的相关结果
        top_items = combined_items[:top_k * 2]
        
        # 从top_items中提取所有的memory_id，确保这些memory会被返回
        top_memory_ids = set()
        # 存储每个memory_id对应的最高fact分数
        memory_fact_scores = {}
        # 保存所有高度相关的fact到字典中，fact_id -> fact
        top_facts_dict = {}
        # 记录每个memory_id直接关联的高度相关fact，memory_id -> [fact]
        memory_direct_facts = {}
        
        for item in top_items:
            if item["type"] == "memory":
                # 对于memory，直接添加其memory_id
                top_memory_ids.add(item["memory_id"])
            elif item["type"] == "fact":
                # 保存高度相关的fact到字典中
                fact = item["item"]
                fact_id = item["fact_id"]
                top_facts_dict[fact_id] = fact
                
                # 对于fact，提取其关联的memory_id
                linked_memory_ids = item.get("linked_memory_ids", [])
                fact_score = item.get("score", 0)
                for mem_id in linked_memory_ids:
                    # 记录每个memory_id对应的最高fact分数
                    if mem_id not in memory_fact_scores or fact_score > memory_fact_scores[mem_id]:
                        memory_fact_scores[mem_id] = fact_score
                    
                    # 记录每个memory_id直接关联的高度相关fact
                    if mem_id not in memory_direct_facts:
                        memory_direct_facts[mem_id] = []
                    memory_direct_facts[mem_id].append(fact)
                    
                    top_memory_ids.add(mem_id)
        
        # 查询所有top_memory_ids对应的memory对象
        all_memories_dict = {}
        if top_memory_ids:
            # 先处理已有的memory（memoryA）
            for memory_id in top_memory_ids:
                if memory_id in memory_dict:
                    all_memories_dict[memory_id] = memory_dict[memory_id]
            
            # 查询剩余的memory_id对应的memoryB
            remaining_memory_ids = top_memory_ids - set(memory_dict.keys())
            if remaining_memory_ids:
                quoted_memory_ids = [f'"{mid}"' for mid in remaining_memory_ids]
                mem_id_filter = f"status == 'active' and user_id == '{user_id}' and memory_id in [{','.join(quoted_memory_ids)}]"
                try:
                    memory_b_res = self.client.query(
                        collection_name=self.mem_col,
                        filter=mem_id_filter,
                        output_fields=["content", "memory_id", "created_at", "user_id"]
                    )
                    
                    # 将memoryB添加到all_memories_dict中
                    for mem_b in memory_b_res:
                        memory_id = mem_b['memory_id']
                        # 计算memoryB与query的内积
                        mem_vec = get_embedding(mem_b["content"])
                        mem_dot_product = sum(a * b for a, b in zip(query_vec, mem_vec))
                        mem_b["original_score"] = mem_dot_product
                        all_memories_dict[memory_id] = mem_b
                except Exception as e:
                    print(f"查询关联记忆失败: {e}")
        
        # ===========================
        # 4. 收集并查询关联memory的关联memory
        # ===========================
        # 从高度相关的fact中收集所有关联的memory_id
        all_linked_memory_ids = set()
        for fact in top_facts_dict.values():
            linked_memory_ids = fact.get("linked_memory_ids", [])
            all_linked_memory_ids.update(linked_memory_ids)
        
        # 查询这些关联memory的其他关联memory
        additional_memory_ids = set()
        if all_linked_memory_ids:
            try:
                # 查询与这些memory关联的所有fact
                # 构建正确的array_contains条件
                array_contains_conditions = []
                for mid in all_linked_memory_ids:
                    array_contains_conditions.append(f'array_contains(linked_memory_ids, "{mid}")')
                
                # 构建完整的过滤条件
                fact_linked_filter = f'user_id == "{user_id}" and ({" or ".join(array_contains_conditions)})'
                linked_facts_res = self.client.query(
                    collection_name=self.fact_col,
                    filter=fact_linked_filter,
                    output_fields=["linked_memory_ids"]
                )
                
                # 收集这些fact关联的所有memory_id，作为additional_memory_ids
                for linked_fact in linked_facts_res:
                    linked_memory_ids = linked_fact.get("linked_memory_ids", [])
                    additional_memory_ids.update(linked_memory_ids)
                
                # 移除已经存在的memory_id
                additional_memory_ids = additional_memory_ids - set(all_memories_dict.keys())
                
                # 查询这些additional_memory_ids对应的memory
                if additional_memory_ids:
                    quoted_additional_ids = [f'"{mid}"' for mid in additional_memory_ids]
                    additional_mem_filter = f"status == 'active' and user_id == '{user_id}' and memory_id in [{','.join(quoted_additional_ids)}]"
                    additional_mem_res = self.client.query(
                        collection_name=self.mem_col,
                        filter=additional_mem_filter,
                        output_fields=["content", "memory_id", "created_at", "user_id"]
                    )
                    
                    # 将additional_memory添加到all_memories_dict中
                    for additional_mem in additional_mem_res:
                        memory_id = additional_mem['memory_id']
                        # 计算additional_memory与query的内积
                        mem_vec = get_embedding(additional_mem["content"])
                        mem_dot_product = sum(a * b for a, b in zip(query_vec, mem_vec))
                        additional_mem["original_score"] = mem_dot_product
                        all_memories_dict[memory_id] = additional_mem
            except Exception as e:
                print(f"查询关联记忆的关联记忆失败: {e}")
        
        # ===========================
        # 5. 处理合并后的记忆
        # ===========================
        results = []
        
        # 收集所有memory关联的其他memory_id
        additional_memory_ids = set()
        
        for memory_id, memory in all_memories_dict.items():
            # ===========================
            # 5.1 处理直接关联的高度相关fact
            # ===========================
            direct_facts = memory_direct_facts.get(memory_id, [])
            direct_fact_ids = set()
            
            # 保存直接关联的高度相关fact
            for fact in direct_facts:
                direct_fact_ids.add(fact.get("fact_id"))
            
            # ===========================
            # 5.2 查询与当前记忆关联的所有其他fact
            # ===========================
            other_facts = []
            try:
                other_facts = self.client.query(
                    collection_name=self.fact_col,
                    filter=f'array_contains(linked_memory_ids, "{memory_id}")',
                    output_fields=["text", "timestamp", "fact_id", "details", "embedding"]
                )
            except Exception as e:
                print(f"查询记忆 {memory_id} 的关联事实失败: {e}")
            
            # ===========================
            # 5.3 合并直接关联的fact和其他fact，避免重复
            # ===========================
            all_related_facts = []
            all_fact_ids = direct_fact_ids.copy()
            
            # 先添加直接关联的高度相关fact
            all_related_facts.extend(direct_facts)
            
            # 再添加其他fact，确保不重复
            for fact in other_facts:
                fact_id = fact.get("fact_id")
                if fact_id not in all_fact_ids:
                    all_fact_ids.add(fact_id)
                    all_related_facts.append(fact)
            
            # ===========================
            # 5.4 计算所有fact与query的内积并排序
            # ===========================
            fact_scores = []
            for fact in all_related_facts:
                try:
                    # 直接使用数据库中存储的embedding，而不是重新计算
                    fact_vec = fact.get("embedding")
                    if not fact_vec or not isinstance(fact_vec, list):
                        # 如果没有embedding字段或不是列表，重新计算，使用text和details拼接
                        fact_vec = self._generate_fact_embedding(fact["text"], fact.get("details", []))
                    
                    # 计算内积
                    dot_product = sum(a * b for a, b in zip(query_vec, fact_vec))
                    fact["similarity"] = dot_product
                    fact_scores.append(dot_product)
                except Exception as e:
                    print(f"计算事实相关性失败: {e}")
                    fact["similarity"] = 0
                    fact_scores.append(0)
            
            # 按内积降序排序
            all_related_facts.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            memory["related_facts"] = all_related_facts[:fact_top_k]
            # 保存事实分数，避免重复计算
            memory["fact_scores"] = fact_scores[:fact_top_k]
            
            # ===========================
            # 5.5 收集当前memory关联的其他memory
            # ===========================
            # 这里可以根据需要从fact或其他地方获取关联memory
            # 目前简单实现，后续可以扩展
            
            # ===========================
            # 5.6 计算综合分数
            # ===========================
            original_score = memory.get("original_score", 0)
            highest_fact_score = memory_fact_scores.get(memory_id, 0)
            
            # 检查是否有直接关联的高度相关fact
            has_direct_fact = memory_id in memory_direct_facts and len(memory_direct_facts[memory_id]) > 0
            
            if has_direct_fact:
                # 如果有直接关联的高度相关fact，给予更高的权重
                # 直接关联的fact对综合分数影响更大
                memory["combined_score"] = (original_score + highest_fact_score * 1.5) / 2
            else:
                # 否则使用默认权重
                memory["combined_score"] = (original_score + highest_fact_score) / 2
            
            # 根据阈值过滤结果
            if memory["combined_score"] >= threshold:
                results.append(memory)
        
        # ===========================
        # 6. 添加关联memory的关联memory
        # ===========================
        # 收集所有memory关联的其他memory
        # 目前简单实现，后续可以扩展
        
        # 根据综合分数重新排序记忆
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # 确保返回的记忆数量不超过top_k
        return results[:top_k]
        
    def _calculate_memory_score(self, memory, enhanced_search=False):
        """直接返回memory与query的内积，不考虑关联事实的相关性"""
        original_score = memory.get("original_score", 0)
        # 直接使用memory与query的内积作为综合分数
        memory["combined_score"] = original_score
        return memory
        
    def _generate_fact_embedding(self, text, details):
        """生成事实的embedding，将text和details拼接起来
        
        Args:
            text: 事实的文本
            details: 事实的详细信息，类型为列表
            
        Returns:
            生成的embedding向量
        """
        # 将details拼接成字符串
        details_str = ""
        if isinstance(details, list) and details:
            # 遍历details列表，将每个details项转换为字符串
            for i, detail in enumerate(details):
                if isinstance(detail, dict):
                    # 如果detail是字典，转换为键值对字符串
                    detail_str = ", ".join([f"{k}: {v}" for k, v in detail.items()])
                    details_str += f"Detail {i+1}: {detail_str}\n"
                else:
                    # 否则直接转换为字符串
                    details_str += f"Detail {i+1}: {str(detail)}\n"
        elif isinstance(details, dict):
            # 如果details是字典，转换为键值对字符串
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        
        # 将text和details拼接成完整的文本
        if details_str:
            full_text = f"{text}\n\nDetails:\n{details_str.strip()}"
        else:
            full_text = text
        
        # 生成embedding
        return get_embedding(full_text)
        
    def generate_response(self, question, question_date, context):
        """生成问题响应"""
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

            # print("#"*50)
            # print("mem:\n", mem)
            # print("#"*50)
            
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
    pipeline = MemoryPipeline(vector_db_type=args.vector_db_type, clear_db=args.clear_db, mode='eval' if args.eval else 'test', dataset_name=args.dataset_type)
    
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
                    lines = json.load(f)
                    if lines and len(lines) > 0:
                        print(f"    共加载 {len(lines)} 条数据")
                        if len(lines) > 0:
                            loaded_item = lines[0]
                            print(f"    第1条数据关键字段：")
                            print(f"      是否包含context：{'context' in loaded_item}")
                            print(f"      是否包含haystack_dates：{'haystack_dates' in loaded_item}")
                            print(f"      数据ID：{loaded_item.get('id', '未知')}")
                    
            # 如果num_users为-1，加载所有数据；否则加载指定数量
            if args.num_users != -1:
                lines = lines[:args.num_users]
            
            # 数据集格式判断和转换
            def is_valid_format(line):
                """判断数据条目是否符合longmemeval格式要求"""
                return "haystack_dates" in line and "haystack_sessions" in line
            
            def convert_hotpotqa_to_expected_format(hotpotqa_item):
                """将HotpotQA条目转换为预期格式"""
                # 生成固定格式的日期
                date = "2023/01/01 (Sun) 12:00"
                
                # 构建系统消息，包含所有背景知识
                context = hotpotqa_item["context"]
                system_content = "以下是背景知识：\n"
                for title, sentences in zip(context["title"], context["sentences"]):
                    system_content += f"\n{title}:\n"
                    for sentence in sentences:
                        system_content += f"- {sentence}\n"
                
                # 构建用户消息，包含问题
                user_content = hotpotqa_item["question"]
                
                # 构建会话结构
                session = [
                    {"role": "system", "content": system_content.strip()},
                    {"role": "user", "content": user_content}
                ]
                
                # 返回转换后的格式，包含question_type字段
                return {
                    "haystack_dates": [date],
                    "haystack_sessions": [session],
                    "id": hotpotqa_item["id"],
                    "answer": hotpotqa_item["answer"],
                    "question_type": hotpotqa_item.get("type", "unknown")  # 使用hotpotqa的type字段作为question_type
                }
            
            # 如果是hotpotqa数据集，检查格式并转换
            if args.dataset_type == "hotpotqa":
                # 检查第一个条目是否符合格式要求
                if lines and not is_valid_format(lines[0]):
                    print(f"HotpotQA数据集格式不符合要求，正在转换 {len(lines)} 个条目...")
                    # 转换所有条目
                    converted_lines = []
                    for i, item in enumerate(lines):
                        if i % 100 == 0:  # 每处理100条打印一次进度
                            print(f"已转换 {i}/{len(lines)} 条数据")
                        converted_lines.append(convert_hotpotqa_to_expected_format(item))
                    lines = converted_lines
                    print(f"转换完成，共转换 {len(lines)} 个条目")
                else:
                    print("HotpotQA数据集格式符合要求，正在检查并确保所有条目包含question_type字段...")
                    # 确保所有条目都包含question_type字段
                    for i, line in enumerate(lines):
                        if "question_type" not in line:
                            # 尝试从原始数据中获取type字段，如果没有则使用默认值
                            lines[i]["question_type"] = line.get("type", "unknown")
                    print("检查完成，所有条目都包含question_type字段")
            
            print(f"已加载 {len(lines)} 个用户/问题。")

            user_detail_results = []
            total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
            
            # 并行处理用户
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交任务 - 确保参数顺序正确：line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type
                # 注意：这里clear_db固定为False，只在主函数中执行一次清空操作
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type): (line, idx) for idx, line in enumerate(lines)}
                
                # 处理结果
                for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
                    line, idx = future_to_user[future]
                    try:
                        result = future.result()
                        user_detail_results.append(result)
                        
                        # 统计总操作次数
                        for key, value in result["counts"].items():
                            total_memory_counts[key] += value
                    except Exception as e:
                        print(f"处理用户 {idx} 时发生错误: {e}")
            
            # 计算总准确率
            correct_count = sum(1 for result in user_detail_results if result["is_correct"])
            accuracy = correct_count / len(user_detail_results) * 100 if user_detail_results else 0
            
            # 按question_type统计每类问题的准确率
            question_type_stats = {}
            for result in user_detail_results:
                q_type = result.get("question_type", "unknown")
                if q_type not in question_type_stats:
                    question_type_stats[q_type] = {"total": 0, "correct": 0}
                question_type_stats[q_type]["total"] += 1
                if result["is_correct"]:
                    question_type_stats[q_type]["correct"] += 1
            
            # 输出评估结果
            print("\n" + "="*50)
            print(f"{args.dataset_type} 评估结果")
            print("="*50)
            print(f"总用户数: {len(user_detail_results)}")
            print(f"正确回答数: {correct_count}")
            print(f"总准确率: {accuracy:.2f}%")
            print(f"记忆操作总数:")
            for op, count in total_memory_counts.items():
                print(f"  {op}: {count}")
            
            # 输出按question_type分类的准确率
            print("\n" + "="*50)
            print("按问题类型分类的准确率")
            print("="*50)
            for q_type, stats in question_type_stats.items():
                type_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"{q_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.2f}%)")
            
            print("="*50)
            
            # 输出详细结果
            print("\n详细结果:")
            for result in user_detail_results:  
                print(f"用户 {result['index']}: {'✓' if result['is_correct'] else '✗'}")
                print(f"  问题: {result['question']}")
                print(f"  问题类型: {result.get('question_type', 'unknown')}")
                print(f"  上下文: {result['context']}")
                print(f"  回答: {result['answer']}...")
                print(f"  标准答案: {result['golden_answer']}...")
                print(f"  记忆操作: {result['counts']}")
                print()
                
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        try:
            # 测试模式
            # 根据数据集类型设置默认数据路径
            if args.dataset_type == "hotpotqa":
                data_path = args.data_path or "./data/hotpotqa-val.jsonl"
            else:  # longmemeval
                data_path = args.data_path or "./data/longmemeval_s_cleaned_test.json"
                
            if not os.path.exists(data_path):
                print(f"数据集文件不存在: {data_path}")
                exit()
            
            # 判断文件类型并加载数据
            lines = []
            if data_path.endswith(".jsonl"):
                # 处理JSONL格式文件
                with open(data_path, "r") as f:
                    for line in f:
                        lines.append(json.loads(line.strip()))
            else:
                # 处理JSON格式文件
                with open(data_path, "r") as f:
                    lines = json.load(f)
                    
            if args.num_users != -1:
                lines = lines[:args.num_users]
            
            print(f"已加载 {len(lines)} 个用户/问题。")

            user_detail_results = []
            total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "INFER": 0, "NOOP": 0}
            
            # 并行处理用户
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交任务，包含extract_mode和dataset_type参数
                future_to_user = {executor.submit(process_and_evaluate_user, line, idx, args.infer, args.retrieve_limit, args.extract_mode, args.vector_db_type, args.dataset_type): (line, idx) for idx, line in enumerate(lines)}
                
                # 处理结果
                for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
                    line, idx = future_to_user[future]
                    try:
                        result = future.result()
                        user_detail_results.append(result)
                        
                        # 统计总操作次数
                        for key, value in result["counts"].items():
                            total_memory_counts[key] += value
                    except Exception as e:
                        print(f"处理用户 {idx} 时发生错误: {e}")
            
            # 计算总准确率
            correct_count = sum(1 for result in user_detail_results if result["is_correct"])
            accuracy = correct_count / len(user_detail_results) * 100 if user_detail_results else 0
            
            # 按question_type统计每类问题的准确率
            question_type_stats = {}
            for result in user_detail_results:
                q_type = result.get("question_type", "unknown")
                if q_type not in question_type_stats:
                    question_type_stats[q_type] = {"total": 0, "correct": 0}
                question_type_stats[q_type]["total"] += 1
                if result["is_correct"]:
                    question_type_stats[q_type]["correct"] += 1
            
            # 输出评估结果
            print("\n" + "="*50)
            print("LongMemEval 评估结果")
            print("="*50)
            print(f"总用户数: {len(user_detail_results)}")
            print(f"正确回答数: {correct_count}")
            print(f"总准确率: {accuracy:.2f}%")
            print(f"记忆操作总数:")
            for op, count in total_memory_counts.items():
                print(f"  {op}: {count}")
            
            # 输出按question_type分类的准确率
            print("\n" + "="*50)
            print("按问题类型分类的准确率")
            print("="*50)
            for q_type, stats in question_type_stats.items():
                type_accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"{q_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.2f}%)")
            
            print("="*50)
            
            # 输出详细结果
            print("\n详细结果:")
            for result in user_detail_results:  
                print(f"用户 {result['index']}: {'✓' if result['is_correct'] else '✗'}")
                print(f"  问题类型: {result.get('question_type', 'unknown')}")
                print(f"  问题: {result['question']}")
                print(f"  上下文: {result['context']}")
                print(f"  回答: {result['answer']}...")
                print(f"  标准答案: {result['golden_answer']}...")
                print(f"  记忆操作: {result['counts']}")
                print()
                
        except Exception as e:
            print(f"评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()