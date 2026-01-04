from copy import deepcopy
import uuid
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from utils import (
    get_embedding, parse_messages, 
    remove_code_blocks, extract_json, get_update_memory_messages,
    LME_JUDGE_MODEL_TEMPLATE, LME_ANSWER_PROMPT, FACT_RETRIEVAL_CORE_MEMORY_TEMPLATE,
    FACT_RETRIEVAL_PROMPT, get_update_memory_messages_core_mem, CORE_MEMORY_UPDATE_PROMPT
)
from lme_eval import lme_grader 
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import pytz
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量
load_dotenv()

# OpenAI 客户端配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 嵌入向量维度
embedding_dim = 1536

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "memory_graph_db"

# 搜索参数配置
topk = 5
search_topk = 10

# 工作记忆最大长度
MAX_WORKING_MEMORY_SIZE = 5

# 初始化 Milvus 连接
def init_milvus_connection():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"成功连接到 Milvus 服务器: {MILVUS_HOST}:{MILVUS_PORT}")
        
        # 定义集合结构
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),  # core, semantic, episodic
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="details", dtype=DataType.JSON),  # 存储详细信息，如 Target商店, 代金券价值等
        ]
        
        schema = CollectionSchema(fields, "记忆图数据库集合")
        
        # 如果集合已存在，删除它
        if utility.has_collection(COLLECTION_NAME):
            print(f"删除已存在的集合: {COLLECTION_NAME}")
            utility.drop_collection(COLLECTION_NAME)
        
        # 创建新集合
        collection = Collection(COLLECTION_NAME, schema)
        print(f"成功创建集合: {COLLECTION_NAME}")
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("成功创建向量索引")
        
        # 加载集合
        collection.load()
        print("集合加载完成")
        
        return collection
    except Exception as e:
        print(f"Milvus 初始化失败: {e}")
        raise

# MemReader Agent 类
class MemReaderAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.working_memory = []  # 工作记忆，存储最近处理的事实和上下文
        self.core_memory = "No core memory yet."
    
    def maintain_working_memory(self, new_item, max_size=MAX_WORKING_MEMORY_SIZE):
        """维护工作记忆，保持最近的交互内容"""
        # 添加新项到工作记忆
        timestamp = datetime.now().isoformat()
        self.working_memory.append({
            "timestamp": timestamp,
            "content": new_item
        })
        
        # 如果工作记忆超过最大长度，移除最早的项
        if len(self.working_memory) > max_size:
            self.working_memory.pop(0)
        
        return self.working_memory
    
    def extract_facts(self, dialogue, core_memory=None):
        """从对话中提取事实，并维护工作记忆"""
        # 使用提供的核心记忆或默认核心记忆
        current_core_memory = core_memory if core_memory else self.core_memory
        
        # 构建包含工作记忆的系统提示
        working_memory_str = "\n".join([
            f"[{item['timestamp']}]: {item['content']}"
            for item in self.working_memory[-3:]
        ])
        
        system_prompt = FACT_RETRIEVAL_CORE_MEMORY_TEMPLATE.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            core_memory=current_core_memory
        )
        
        # 添加工作记忆到提示
        if working_memory_str:
            system_prompt += f"\n\nRecent Working Memory:\n{working_memory_str}"
        
        user_prompt = f"Input:\n{dialogue}"
        
        # 调用 LLM 提取事实
        llm_response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )
        
        response = llm_response.choices[0].message.content
        
        # 更新工作记忆
        self.maintain_working_memory(dialogue)
        
        # 解析提取的事实
        if response == '{"facts" : []}':
            return []
        
        try:
            response = remove_code_blocks(response)
            if not response.strip():
                return []
            
            try:
                extracted_data = json.loads(response)
                facts = extracted_data.get("facts", [])
                
                # 更新工作记忆中的事实
                for fact in facts:
                    self.maintain_working_memory(f"Extracted fact: {fact}")
                
                return facts
            except json.JSONDecodeError:
                extracted_json = extract_json(response)
                extracted_data = json.loads(extracted_json)
                facts = extracted_data.get("facts", [])
                
                # 更新工作记忆中的事实
                for fact in facts:
                    self.maintain_working_memory(f"Extracted fact: {fact}")
                
                return facts
        except Exception as e:
            print(f"提取事实时出错: {e}")
            return []
    
    def extract_fact_details(self, facts):
        """从提取的事实中提取更详细的信息（如实体、属性、关系等）"""
        detailed_facts = []
        
        for fact in facts:
            try:
                # 构建提示来提取事实的详细信息
                prompt = f"""
请从以下事实中提取详细信息：

事实：{fact}

请识别：
1. 主要实体/主题（target）
2. 该实体的属性/特性
3. 与其他实体的关系（如果有）
4. 任何特定的值、日期或关键细节
5. 这个事实的核心含义

请以JSON格式返回结果，包含以下键：
- target: 主要实体/主题
- details: 表示属性和值的键值对数组
"""
                
                response = self.llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                
                fact_details = response.choices[0].message.content
                fact_details = json.loads(fact_details)
                
                # 确保格式正确
                if "target" not in fact_details:
                    fact_details["target"] = fact
                if "details" not in fact_details:
                    fact_details["details"] = []
                
                # 添加原始事实
                fact_details["original_fact"] = fact
                detailed_facts.append(fact_details)
                
                # 更新工作记忆
                self.maintain_working_memory(f"Detailed fact: {fact_details['target']} -> {fact_details['details']}")
                
            except Exception as e:
                print(f"提取事实详情时出错: {e}")
                # 如果出错，使用简化版本
                detailed_facts.append({
                    "target": fact,
                    "details": [],
                    "original_fact": fact
                })
        
        return detailed_facts
    
    def update_core_memory(self, dialogue, current_core_memory=None):
        """更新核心记忆"""
        current_core = current_core_memory if current_core_memory else self.core_memory
        
        try:
            core_prompt = CORE_MEMORY_UPDATE_PROMPT.format(
                core_memory=current_core,
                new_dialogue=dialogue
            )
            
            core_response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": core_prompt}], 
                temperature=0, 
            )
            
            updated_core = core_response.choices[0].message.content.strip()
            if updated_core and updated_core != "No core memory yet.":
                self.core_memory = updated_core
                # 更新工作记忆
                self.maintain_working_memory(f"Updated core memory: {updated_core[:100]}...")
                return updated_core
        except Exception as e:
            print(f"更新核心记忆时出错: {e}")
        
        return current_core

# Memory Manager 类，负责与 Milvus 图数据库交互
class MemoryManager:
    def __init__(self, collection, llm_client):
        self.collection = collection
        self.llm_client = llm_client
    
    def get_embedding(self, text):
        """获取文本的嵌入向量"""
        try:
            # 使用OpenAI获取嵌入向量
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}")
            return None
    
    def search_memory(self, query_text, memory_type=None, top_k=5):
        """在记忆库中搜索相关记忆"""
        try:
            query_embedding = self.get_embedding(query_text)
            if query_embedding is None:
                return []
            
            # 构建搜索表达式
            expr = None
            if memory_type:
                expr = f"memory_type == '{memory_type}'"
            
            # 优化的搜索参数
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "content", "memory_type", "created_at", "updated_at", "details"]
            )
            
            # 格式化搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content"),
                        "text": hit.entity.get("content"),  # 添加text字段以兼容其他组件
                        "memory_type": hit.entity.get("memory_type"),
                        "score": hit.score,
                        "created_at": hit.entity.get("created_at"),
                        "updated_at": hit.entity.get("updated_at"),
                        "details": hit.entity.get("details", {})
                    })
            
            return search_results
        except Exception as e:
            print(f"搜索记忆时出错: {e}")
            return []
    
    def add_memory(self, content, memory_type="semantic", details=None):
        """添加新记忆到数据库"""
        try:
            embedding_vector = self.get_embedding(content)
            if embedding_vector is None:
                return None
            
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # 准备插入数据
            data = [
                [memory_id],  # id
                [embedding_vector],  # embedding
                [content],  # content
                [memory_type],  # memory_type
                [timestamp],  # created_at
                [timestamp],  # updated_at
                [details if details else {}]  # details
            ]
            
            # 插入数据
            self.collection.insert(data)
            # 刷新集合以确保数据可搜索
            self.collection.flush()
            print(f"成功添加记忆: {memory_id} - {content[:50]}...")
            
            return {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "created_at": timestamp,
                "details": details
            }
        except Exception as e:
            print(f"添加记忆时出错: {e}")
            return None
    
    def update_memory(self, memory_id, new_content, details=None):
        """更新现有记忆"""
        try:
            # 获取现有记忆
            expr = f"id == '{memory_id}'"
            results = self.collection.query(expr=expr, output_fields=["id", "content", "memory_type", "created_at", "details"])
            
            if not results:
                print(f"未找到要更新的记忆: {memory_id}")
                return None
            
            existing_memory = results[0]
            embedding_vector = self.get_embedding(new_content)
            if embedding_vector is None:
                return None
            
            # 更新时间戳
            updated_at = datetime.now().isoformat()
            
            # 合并详细信息
            updated_details = existing_memory.get("details", {})
            if details:
                updated_details.update(details)
            
            # 准备更新数据
            data = [
                [memory_id],  # id
                [embedding_vector],  # embedding
                [new_content],  # content
                [existing_memory["memory_type"]],  # memory_type
                [existing_memory["created_at"]],  # created_at
                [updated_at],  # updated_at
                [updated_details]  # details
            ]
            
            # 删除旧数据
            self.collection.delete(expr=expr)
            
            # 插入更新后的数据
            self.collection.insert(data)
            # 刷新集合
            self.collection.flush()
            print(f"成功更新记忆: {memory_id} - {new_content[:50]}...")
            
            return {
                "id": memory_id,
                "content": new_content,
                "memory_type": existing_memory["memory_type"],
                "created_at": existing_memory["created_at"],
                "updated_at": updated_at,
                "details": updated_details
            }
        except Exception as e:
            print(f"更新记忆时出错: {e}")
            return None
    
    def delete_memory(self, memory_id):
        """删除记忆"""
        try:
            expr = f"id == '{memory_id}'"
            result = self.collection.delete(expr=expr)
            # 刷新集合
            self.collection.flush()
            print(f"成功删除记忆: {memory_id}")
            return True
        except Exception as e:
            print(f"删除记忆时出错: {e}")
            return False
    
    def judge_memory_action(self, detailed_facts, existing_memories, core_memory):
        """判断对记忆的操作类型（ADD、UPDATE、DELETE）"""
        try:
            # 准备现有记忆的格式
            retrieved_old_memory_dict = []
            memory_id_mapping = {}
            
            for idx, mem in enumerate(existing_memories):
                mem_id = str(idx)
                retrieved_old_memory_dict.append({
                    "id": mem_id,
                    "text": mem["content"]
                })
                memory_id_mapping[mem_id] = mem["id"]
            
            # 构建优化的提示以判断记忆操作
            prompt = f"""
            你是一个记忆管理专家，需要分析新提取的事实和已有的记忆，决定对每个新事实应该执行什么操作。
            
            Core Memory: {core_memory}
            
            已有的相关记忆：
            {json.dumps(retrieved_old_memory_dict, ensure_ascii=False, indent=2)}
            
            新提取的详细事实：
            {json.dumps(detailed_facts, ensure_ascii=False, indent=2)}
            
            对于每个新提取的事实，请决定执行以下操作之一：
            1. ADD: 如果这是一个全新的事实，与现有记忆没有显著重叠
            2. UPDATE: 如果这个事实与某个现有记忆高度相关，但需要更新或合并
            3. DELETE: 如果现有记忆不准确或过时，需要被这个新事实替换
            4. NONE: 如果这个事实已经完全包含在现有记忆中，不需要任何操作
            
            请以JSON格式返回结果，包含以下结构：
            {
              "memory": [
                {
                  "id": "现有记忆的ID（如果操作是UPDATE或DELETE），否则为null",
                  "text": "新事实的内容",
                  "event": "ADD/UPDATE/DELETE/NONE",
                  "old_memory": "如果是UPDATE或DELETE，对应的旧记忆内容",
                  "details": "事实的详细信息"
                }
              ],
              "core_memory": "更新后的核心记忆（如果需要）"
            }
            """
            
            # 调用LLM进行判断
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            
            # 解析响应
            update_response = response.choices[0].message.content
            response_str = remove_code_blocks(update_response)
            response_json = json.loads(response_str)
            
            # 提取核心记忆更新
            updated_core_memory = response_json.get("core_memory", core_memory)
            
            # 提取记忆操作
            memory_actions = response_json.get("memory", [])
            
            # 映射回实际的记忆ID
            actions_with_actual_ids = []
            for action in memory_actions:
                mem_id = action.get("id")
                if mem_id in memory_id_mapping:
                    action["actual_id"] = memory_id_mapping[mem_id]
                actions_with_actual_ids.append(action)
            
            return updated_core_memory, actions_with_actual_ids
        
        except Exception as e:
            print(f"判断记忆操作时出错: {e}")
            return core_memory, []

# 初始化 Milvus 集合
try:
    milvus_collection = init_milvus_connection()
except Exception as e:
    print(f"初始化 Milvus 失败，程序退出: {e}")
    exit(1)

# Qdrant相关函数已移除，改用MemoryManager类中的Milvus操作方法

def generate_response(llm_client, question, question_date, context, core_memory=""):
    full_context = f"{context}\nUser Profile (Core Memory):\n{core_memory}"
    
    prompt = LME_ANSWER_PROMPT.format(
        question=question,
        question_date=question_date,
        context=full_context
    )
    response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                temperature=0,
            )
    return response, full_context


 
def process_user_memory_infer(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    
    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    current_core_memory = "No core memory yet."
    
    # 初始化MemReaderAgent和MemoryManager
    mem_reader_agent = MemReaderAgent(llm_client=openai_client)
    memory_manager = MemoryManager(collection=milvus_collection, llm_client=openai_client)

    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        for turn_id in range(0, len(session), 2):
            # 解析对话
            parsed_messages = parse_messages(session[turn_id:turn_id+2])
            print("="*40)
            print("parsed_messages:", parsed_messages)
            print("="*40)
            
            # Step 1: 使用MemReaderAgent提取事实并维护工作记忆
            # 1.1 提取事实
            new_retrieved_facts = mem_reader_agent.extract_facts(parsed_messages, core_memory=current_core_memory)
            print(f"新检索到的事实: {new_retrieved_facts}")
            
            # 1.2 更新工作记忆
            mem_reader_agent.maintain_working_memory(new_retrieved_facts)
            
            # 1.3 提取事实详情
            detailed_facts = mem_reader_agent.extract_fact_details(new_retrieved_facts)
            print(f"提取的事实详情: {detailed_facts}")
            
            # 1.4 更新核心记忆
            updated_core_memory = mem_reader_agent.update_core_memory(parsed_messages, current_core_memory=current_core_memory)
            if updated_core_memory and updated_core_memory != "No core memory yet.":
                current_core_memory = updated_core_memory
                print(f"更新后的核心记忆: {current_core_memory}")
            
            # 如果没有提取到事实，跳过后续处理
            if not new_retrieved_facts or not detailed_facts:
                continue
            
            # Step 2: 使用MemoryManager搜索相关记忆
            # 2.1 搜索与提取事实相关的现有记忆
            retrieved_old_facts = []
            # 对每个详细事实进行搜索
            for detailed_fact in detailed_facts:
                search_text = detailed_fact.get("target", detailed_fact.get("original_fact", ""))
                if search_text:
                    related_memories = memory_manager.search_memory(search_text, top_k=3)
                    for mem in related_memories:
                        # 避免重复添加
                        if not any(existing_mem.get("id") == mem.get("id") for existing_mem in retrieved_old_facts):
                            retrieved_old_facts.append({
                                "id": mem.get("id"),
                                "text": mem.get("text", mem.get("content", "")),
                                "memory_type": mem.get("memory_type", "semantic"),
                                "details": mem.get("details", {})
                            })
            
            print(f"检索到的现有记忆: {retrieved_old_facts}")
            
            # Step 3: 使用MemoryManager判断记忆操作
            # 3.1 判断应该执行的记忆操作
            memory_action_result = memory_manager.judge_memory_action(
                detailed_facts=detailed_facts,
                existing_memories=retrieved_old_facts,
                core_memory=current_core_memory
            )
            
            # 3.2 更新核心记忆（如果返回了新的核心记忆）
            if "core_memory" in memory_action_result and memory_action_result["core_memory"]:
                current_core_memory = memory_action_result["core_memory"]
            
            # Step 4: 执行记忆操作
            memory_operations = memory_action_result.get("memory", [])
            for mem_op in memory_operations:
                event_type = mem_op.get("event")
                action_text = mem_op.get("text")
                memory_id = mem_op.get("id")
                details = mem_op.get("details", {})
                
                if event_type in operation_counts:
                    operation_counts[event_type] += 1
                
                # 执行相应的记忆操作
                if event_type == "ADD" and action_text:
                    # 添加新记忆到Milvus
                    new_memory_id = memory_manager.add_memory(
                        content=action_text,
                        memory_type="semantic",
                        details=details
                    )
                    if new_memory_id:
                        print(f"成功添加记忆: {new_memory_id} - {action_text[:50]}...")
                
                elif event_type == "UPDATE" and memory_id and action_text:
                    # 更新现有记忆
                    success = memory_manager.update_memory(
                        memory_id=memory_id,
                        new_content=action_text,
                        details=details
                    )
                    if success:
                        print(f"成功更新记忆: {memory_id} - {action_text[:50]}...")
                
                elif event_type == "DELETE" and memory_id:
                    # 删除记忆
                    success = memory_manager.delete_memory(memory_id)
                    if success:
                        print(f"成功删除记忆: {memory_id}")
                
                elif event_type == "NONE":
                    print(f"无需操作: {action_text}")
            
            # 记录工作记忆状态
            print(f"当前工作记忆: {mem_reader_agent.working_memory}")
    
    # 返回操作统计和最新的核心记忆
    return operation_counts, current_core_memory


def process_user_memory(line):
    dates = line.get("haystack_dates")
    sessions = line.get("haystack_sessions")
    
    operation_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    current_core_memory = "No core memory yet."
    
    # 初始化MemReaderAgent和MemoryManager
    mem_reader_agent = MemReaderAgent(llm_client=openai_client)
    memory_manager = MemoryManager(collection=milvus_collection, llm_client=openai_client)

    for session_id, session in enumerate(sessions):
        date = dates[session_id] + " UTC"
        date_format = "%Y/%m/%d (%a) %H:%M UTC"
        date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
        
        for turn_id in range(0, len(session), 2):
            # 解析对话
            parsed_messages = parse_messages(session[turn_id:turn_id+2])
            print("="*40)
            print("parsed_messages:", parsed_messages)
            print("="*40)
            
            # Step 1: 使用MemReaderAgent提取事实并维护工作记忆
            # 1.1 提取事实
            new_retrieved_facts = mem_reader_agent.extract_facts(parsed_messages, core_memory=current_core_memory)
            print(f"新检索到的事实: {new_retrieved_facts}")
            
            # 1.2 更新工作记忆
            mem_reader_agent.maintain_working_memory(new_retrieved_facts)
            
            # 1.3 提取事实详情（保持简单模式的实现）
            detailed_facts = mem_reader_agent.extract_fact_details(new_retrieved_facts)
            print(f"提取的事实详情: {detailed_facts}")
            
            # 1.4 更新核心记忆
            updated_core_memory = mem_reader_agent.update_core_memory(parsed_messages, current_core_memory=current_core_memory)
            if updated_core_memory and updated_core_memory != "No core memory yet.":
                current_core_memory = updated_core_memory
                print(f"更新后的核心记忆: {current_core_memory}")
            
            # 如果没有提取到事实，跳过后续处理
            if not new_retrieved_facts or not detailed_facts:
                continue
            
            # Step 2: 简化模式 - 直接将所有事实添加到Milvus（不进行UPDATE/DELETE判断）
            # 这保持了原函数的简单逻辑，只进行ADD操作
            for fact in new_retrieved_facts:
                try:
                    # 为每个事实创建详细信息
                    fact_detail = next((d for d in detailed_facts if d.get("original_fact") == fact), None)
                    details = fact_detail.get("details", {}) if fact_detail else {}
                    
                    # 添加记忆到Milvus
                    memory_manager.add_memory(
                        content=fact,
                        memory_type="semantic",
                        details={
                            "original_fact": fact,
                            "extracted_details": details,
                            "created_at": date_string.isoformat()
                        }
                    )
                    operation_counts["ADD"] += 1
                    print(f"成功添加事实到Milvus: {fact[:50]}...")
                except Exception as e:
                    print(f"添加事实时出错: {e}")
            
            # 记录工作记忆状态
            print(f"当前工作记忆: {mem_reader_agent.working_memory}")
    
    # 返回操作统计和最新的核心记忆
    return operation_counts, current_core_memory

def response_user(line, core_memory):
    question = line.get("question")
    question_date = line.get("question_date")
    question_date = question_date + " UTC"
    question_date_format = "%Y/%m/%d (%a) %H:%M UTC"
    question_date_string = datetime.strptime(question_date, question_date_format).replace(tzinfo=timezone.utc)
    
    # 初始化MemoryManager用于检索记忆
    memory_manager = MemoryManager(collection=milvus_collection, llm_client=openai_client)
    
    # 从Milvus检索相关记忆
    retrieved_memories = memory_manager.search_memory(query_text=question, top_k=search_topk)
    
    # 格式化检索到的记忆为字符串
    memories_str = ""
    if retrieved_memories:
        memories_str = "\n".join([
            f"- {mem.get('created_at', 'Unknown time')}: {mem.get('content', '')}"
            for mem in retrieved_memories
        ])
    else:
        memories_str = "没有检索到相关记忆"
    
    # 传入Core Memory生成回答
    response, full_context = generate_response(openai_client, question, question_date_string, memories_str, core_memory)
    answer = response.choices[0].message.content

    return full_context, answer

def process_and_evaluate_user(line, user_index, client, infer):
    try:
        if infer:
            # 调用更新后的 process_user_memory_infer 函数
            result = process_user_memory_infer(line)
            memory_counts = result.get('memory_count', {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0})
            core_memory = result.get('core_memory', {})
        else:
            # 调用更新后的 process_user_memory 函数
            result = process_user_memory(line)
            memory_counts = result.get('memory_count', {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0})
            core_memory = result.get('core_memory', {})
        
        # 将 Core Memory 传入 response_user
        full_context, answer = response_user(line, core_memory)
        
        golden_answer = line.get("answer") 
        question = line.get("question")
        
        # 使用 lme_grader 评估回答正确性
        is_correct = False
        try:
            is_correct = lme_grader(client, question, golden_answer, answer)
        except Exception as grader_error:
            print(f"使用 lme_grader 评估时出错: {str(grader_error)}")
        
        # 返回处理结果
        return {
            "index": user_index,
            "is_correct": is_correct,
            "counts": memory_counts,
            "core_memory": core_memory, # 记录
            "retrieved_memories": full_context,
            "question": question,
            "answer": answer,
            "golden_answer": golden_answer
        }
    except Exception as e:
        print(f"Error processing user {user_index}: {e}")
        # 返回错误情况下的基本信息
        return {
            "index": user_index,
            "is_correct": False,
            "counts": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0},
            "core_memory": "No core memory yet.",
            "question": line.get("question", "N/A")
        }

if __name__ == "__main__":
    # 主程序入口 - 使用 Milvus 作为图数据库
    # Milvus 已在文件开头通过 init_milvus_connection() 初始化
    print(f"已成功连接到 Milvus 服务器: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"使用集合: {COLLECTION_NAME}")
    
    # 可选：清空 Milvus 集合中的数据（如果需要）
    try:
        # 确保集合已加载
        milvus_collection.load()
        # 删除所有数据
        expr = ""
        milvus_collection.delete(expr)
        print("已清空 Milvus 集合中的所有数据")
    except Exception as e:
        print(f"清空 Milvus 数据时发生异常（继续执行）: {e}")

    with open("./data/longmemeval_s_cleaned.json", "r") as f:
        lines = json.load(f)[:50]
    
    print(f"已加载 {len(lines)} 个用户/问题。")

    user_detail_results = [] 
    total_memory_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
    
    MAX_WORKERS = 10
    infer = False

    futures = []

    print(f"开始使用 {MAX_WORKERS} 个线程并行处理...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, line in enumerate(lines):
            future = executor.submit(process_and_evaluate_user, line, idx + 1, openai_client, infer=infer)
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
            result = future.result()
            user_detail_results.append(result)

    user_detail_results.sort(key=lambda x: x.get("index", 0))

    correct_count = 0
    total_evaluated = len(user_detail_results)

    for res in user_detail_results:
        if res.get("is_correct"):
            correct_count += 1
        
        counts = res.get("counts", {})
        for key in total_memory_counts:
            total_memory_counts[key] += counts.get(key, 0)

    print("\n\n==================================================")
    print("              🎯 最终评估结果") 
    print("==================================================")

    if total_evaluated > 0:
        final_accuracy = correct_count / total_evaluated
        print(f"总评估问题数: {total_evaluated}")
        print(f"正确回答数: {correct_count}")
        print(f"最终总准确率: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    else:
        print("没有评估任何问题。")
    print("==================================================")

    print("\n\n==================================================")
    print("        📊 详细记忆操作统计 (按用户)")
    print("==================================================")

    for res in user_detail_results:
        user_index = res["index"]
        is_correct = res["is_correct"]
        counts = res["counts"]
        question = res["question"]
        answer = res.get("answer", "N/A")
        golden_answer = res.get("golden_answer", "N/A")
        core_mem = res.get("core_memory", "N/A")
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        print(f"\n--- 用户/问题 {user_index} ---")
        print(f"  问题: {question}...")
        # print(f"  Core Memory: {core_mem}") 
        print(f"  检索记忆: {res.get('retrieved_memories', '')}...")
        print(f"  模型回答: {answer}...")
        print(f"  标准答案: {golden_answer}...")
        print(f"  评估结果: {status}")
        print(f"  记忆操作: ADD={counts.get('ADD', 0)}, UPDATE={counts.get('UPDATE', 0)}, DELETE={counts.get('DELETE', 0)}, NONE={counts.get('NONE', 0)}")

    print("\n--- 所有用户的记忆操作总览 ---")
    print(f"  ADD (新增):    {total_memory_counts['ADD']}")
    print(f"  UPDATE (更新): {total_memory_counts['UPDATE']}")
    print(f"  DELETE (删除): {total_memory_counts['DELETE']}")
    print(f"  NONE (无操作): {total_memory_counts['NONE']}")
    print("==================================================")