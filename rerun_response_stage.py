import os
import json
import argparse
import time
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline_chat_history_memory import MemoryPipeline, response_user, get_embedding, llm_client

# 从 pipeline_chat_history 导入必要的组件
# 确保环境配置已加载
from dotenv import load_dotenv
load_dotenv()

def process_response_only_user(line, user_index, retrieve_limit: int = 3, vector_db_type="milvus", dataset_name="", threshold=0.0):
    """
    仅执行响应阶段的评估，假设记忆库已构建。
    """
    try:
        # 为每个用户生成唯一的user_id，确保记忆隔离
        user_id = f"user_{user_index}"
        
        # 为每个用户创建独立的pipeline实例
        pipeline = MemoryPipeline(vector_db_type=vector_db_type, clear_db=False, dataset_name=dataset_name)
        
        # 直接进入生成问题响应阶段，传递user_id
        # 这里使用 search_memories 内部的相似度计算逻辑
        retrieved_memories, answer = response_user(line, pipeline, retrieve_limit, user_id=user_id, threshold=threshold)
        
        # 确保retrieved_memories不是None
        retrieved_memories = retrieved_memories or []
        
        # 构建上下文字符串用于展示结果
        memories_with_facts = []
        for mem in retrieved_memories:
            memory_line = f"- [{datetime.fromtimestamp(mem['created_at'], timezone.utc).isoformat()}] {mem['content']}"
            memories_with_facts.append(memory_line)
            
            related_facts = mem.get("related_facts", [])
            for i, fact in enumerate(related_facts[:3]):
                fact_text = fact['text']
                fact_timestamp = fact.get('timestamp')
                timestamp_str = f"[{datetime.fromtimestamp(fact_timestamp, timezone.utc).isoformat()}] " if fact_timestamp else ""
                memories_with_facts.append(f"  ├── [{i+1}] {timestamp_str}事实: {fact_text}")
        
        memories_str = "\n".join(memories_with_facts)
        
        # 获取标准答案和问题类型
        golden_answer = line.get("answer", "N/A")
        question_type = line.get("question_type", "unknown")
        
        # 评估答案准确性
        from lme_eval import lme_grader
        question = line.get("question", "")
        is_correct = lme_grader(llm_client, question, golden_answer, answer)
        
        return {
            "index": user_index,
            "is_correct": is_correct,
            "question": line.get("question"),
            "question_type": question_type,
            "context": memories_str,
            "answer": answer,
            "golden_answer": golden_answer
        }
    except Exception as e:
        import traceback
        print(f"处理用户 {user_index} 时发生错误: {e}")
        traceback.print_exc()
        return {
            "index": user_index,
            "is_correct": False,
            "question": line.get("question", "N/A"),
            "question_type": line.get("question_type", "unknown"),
            "context": "N/A (Error)",
            "answer": "N/A",
            "golden_answer": line.get("answer", "N/A")
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun Response Stage Only (Memory DB must be pre-built)")
    parser.add_argument("--num_users", type=int, default=50, help="评估用户数量")
    parser.add_argument("--max_workers", type=int, default=10, help="并行处理的工作线程数")
    parser.add_argument("--retrieve_limit", type=int, default=3, help="检索时返回的记忆数量")
    parser.add_argument("--threshold", type=float, default=0.0, help="记忆相似度阈值")
    parser.add_argument("--vector-db-type", type=str, default="milvus", choices=["milvus", "qdrant"], help="向量数据库类型")
    parser.add_argument("--data-path", type=str, help="指定数据文件路径")
    parser.add_argument("--dataset-type", type=str, default="longmemeval", choices=["longmemeval", "hotpotqa"], help="数据集类型")
    parser.add_argument("--model", type=str, help="指定调用的 LLM 模型名称 (例如 gemini-3-pro-preview)")
    args = parser.parse_args()
    
    # 如果指定了模型，设置环境变量以便 pipeline_chat_history 使用
    
    # 根据数据集类型设置默认数据路径
    if args.dataset_type == "hotpotqa":
        data_path = args.data_path or "./data/hotpotqa-val.jsonl"
    else:
        data_path = args.data_path or "./data/lme/longmemeval_s_cleaned.json"
        
    if not os.path.exists(data_path):
        print(f"数据集文件不存在: {data_path}")
        exit()
        
    # 加载数据
    if data_path.endswith(".jsonl"):
        with open(data_path, "r") as f:
            lines = [json.loads(l.strip()) for l in f]
    else:
        with open(data_path, "r") as f:
            lines = json.load(f)
            
    if args.num_users != -1:
        lines = lines[:args.num_users]
        
    print(f"开始重跑 Response 阶段，共 {len(lines)} 个用户/问题...")
    
    user_detail_results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_user = {
            executor.submit(
                process_response_only_user, 
                line, idx, args.retrieve_limit, args.vector_db_type, args.dataset_type, args.threshold
            ): idx for idx, line in enumerate(lines)
        }
        
        for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
            result = future.result()
            user_detail_results.append(result)
            
    # 统计结果
    correct_count = sum(1 for r in user_detail_results if r["is_correct"])
    accuracy = correct_count / len(user_detail_results) * 100 if user_detail_results else 0
    
    question_type_stats = {}
    for result in user_detail_results:
        q_type = result.get("question_type", "unknown")
        if q_type not in question_type_stats:
            question_type_stats[q_type] = {"total": 0, "correct": 0}
        question_type_stats[q_type]["total"] += 1
        if result["is_correct"]:
            question_type_stats[q_type]["correct"] += 1
            
    # 输出结果
    print("\n" + "="*50)
    print(f"Rerun Response Stage 结果 ({args.dataset_type})")
    print("="*50)
    print(f"总用户数: {len(user_detail_results)}")
    print(f"正确回答数: {correct_count}")
    print(f"总准确率: {accuracy:.2f}%")
    
    print("\n按问题类型分类的准确率:")
    for q_type, stats in question_type_stats.items():
        type_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {q_type}: {stats['correct']}/{stats['total']} ({type_acc:.2f}%)")
    
    # === 新增：详细结果打印（模仿原脚本输出） ===
    print("\n" + "="*50)
    print("详细检索与回答结果:")
    print("="*50)
    # 按索引排序保证输出顺序
    user_detail_results.sort(key=lambda x: x["index"])
    for result in user_detail_results:
        print(f"\n用户 {result['index']}: {'✓' if result['is_correct'] else '✗'}")
        print(f"  问题类型: {result.get('question_type', 'unknown')}")
        print(f"  问题: {result['question']}")
        print(f"  上下文:")
        # 缩减打印上下文内容，保持缩进
        for line in result['context'].split('\n'):
            print(f"  {line}")
        print(f"  回答: {result['answer']}")
        print(f"  标准答案: {result['golden_answer']}")
        # 由于是重跑响应阶段，记忆操作统计通常为0或不适用，这里留空或标注
        # print(f"  记忆操作: 不适用 (重跑模式)")

    # 保存详细结果到文件
    output_file = f"rerun_results_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(user_detail_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_file}")
