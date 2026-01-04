#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于新EVAL_PROMPT评估抽取事实质量的脚本
评估内容：
1. Accuracy: 事实是否严格支持对话？
2. Self-Containment: 事实能否脱离对话历史被理解？代词是否已解析？
3. Format Compliance: details是否为"Category: Value"格式？除非必要，否则排除日期？
4. Worthiness: 这是有用的长期信息（偏好、计划、事实）还是闲聊？
"""

import json
import argparse
from typing import Dict, List, Tuple, Any
from openai import OpenAI
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class NewFactQualityEvaluator:
    def __init__(self, model_name: str = "gpt-4.1"):
        """初始化评估器"""
        self.client = OpenAI()
        self.model_name = model_name
        self.init_prompts()
    
    def init_prompts(self):
        """初始化评估任务的prompt模板"""
        self.EVAL_PROMPT = """ 
 You are a Quality Assurance Auditor for a Personal Memory System. 
 Your task is to evaluate the quality of "Facts" extracted from a "Conversation". 
 
 ### Input Data 
 Conversation: 
 {conversation_text} 
 
 Extracted Fact: 
 {extracted_json} 
 
 ### Evaluation Criteria 
 1. **Accuracy**: Is the fact strictly supported by the conversation? (Yes/No) 
 2. **Self-Containment**: Can the fact be understood without the conversation history? Are pronouns resolved? (Yes/No) 
 3. **Format Compliance**: Are details in "Category: Value" format? Are dates excluded unless necessary? (Yes/No) 
 4. **Worthiness**: Is this useful long-term information (preferences, plans, facts) rather than chit-chat? (Yes/No) 
 
 ### Output Format 
 Return a JSON object: 
 {{ 
   "accuracy_score": 1, 
   "self_containment_score": 1, 
   "format_compliance_score": 1, 
   "worthiness_score": 1, 
   "reasoning": "Explain why it passed or failed..." 
 }} 
 """
    
    def _get_llm_response(self, prompt: str, input_data: Dict) -> Dict:
        """获取LLM响应"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # 格式化提示
                formatted_prompt = prompt.format(**input_data)
                
                # 调用OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Quality Assurance Auditor for a Personal Memory System."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                response_content = response.choices[0].message.content.strip()
                # 解析JSON响应
                return json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"原始响应: {response_content[:200]}...")
            except Exception as e:
                print(f"LLM调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                print(f"{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
        
        # 如果所有重试都失败，返回默认值
        return {
            "accuracy_score": 0,
            "self_containment_score": 0,
            "format_compliance_score": 0,
            "worthiness_score": 0,
            "reasoning": "Failed to get valid evaluation response after multiple attempts"
        }
    
    def evaluate_fact(self, conversation_text: str, fact: Dict) -> Dict:
        """评估单个事实"""
        # 准备输入数据
        input_data = {
            "conversation_text": conversation_text,
            "extracted_json": json.dumps(fact, ensure_ascii=False)
        }
        
        # 获取评估结果
        evaluation_result = self._get_llm_response(self.EVAL_PROMPT, input_data)
        return evaluation_result
    
    def _process_entry(self, entry: Dict, entry_idx: int) -> Dict:
        """处理单个条目，返回条目结果"""
        entry_result = {
            "entry_id": entry.get("id", entry_idx),
            "chunk_results": [],
            "overall_scores": {
                "average_accuracy": 0,
                "average_self_containment": 0,
                "average_format_compliance": 0,
                "average_worthiness": 0,
                "average_overall": 0
            }
        }
        
        chunks = entry.get("chunks", [])
        facts_of_chunks = entry.get("facts_of_chunks", [])
        
        # 确保chunks和facts_of_chunks长度匹配
        min_len = min(len(chunks), len(facts_of_chunks))
        
        # 收集所有评分，用于计算平均值
        all_scores = []
        
        # 处理每个chunk及其对应的facts
        for chunk_idx in range(min_len):
            chunk = chunks[chunk_idx]
            chunk_facts = facts_of_chunks[chunk_idx] if chunk_idx < len(facts_of_chunks) else []
            
            chunk_result = {
                "chunk_idx": chunk_idx,
                "fact_evaluations": []
            }
            
            # 评估该chunk中的每个fact
            for fact_idx, fact in enumerate(chunk_facts):
                evaluation = self.evaluate_fact(chunk, fact)
                evaluation["fact_idx"] = fact_idx
                evaluation["fact"] = fact
                
                chunk_result["fact_evaluations"].append(evaluation)
                
                # 收集评分
                scores = [
                    evaluation["accuracy_score"],
                    evaluation["self_containment_score"],
                    evaluation["format_compliance_score"],
                    evaluation["worthiness_score"]
                ]
                all_scores.append(scores)
            
            entry_result["chunk_results"].append(chunk_result)
        
        # 计算整体平均分
        if all_scores:
            total_scores = [sum(category) for category in zip(*all_scores)]
            avg_scores = [score / len(all_scores) for score in total_scores]
            
            entry_result["overall_scores"] = {
                "average_accuracy": avg_scores[0],
                "average_self_containment": avg_scores[1],
                "average_format_compliance": avg_scores[2],
                "average_worthiness": avg_scores[3],
                "average_overall": sum(avg_scores) / 4
            }
        
        return entry_result
    
    def evaluate_dataset(self, json_path: str) -> Dict:
        """评估整个数据集"""
        print(f"正在读取JSON文件: {json_path}")
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 个条目")
        
        # 初始化结果
        results = {
            "total_entries": len(data),
            "entries_evaluated": 0,
            "entry_results": [],
            "overall_statistics": {
                "average_accuracy": 0,
                "average_self_containment": 0,
                "average_format_compliance": 0,
                "average_worthiness": 0,
                "average_overall": 0,
                "total_facts_evaluated": 0,
                "facts_passed_all_criteria": 0
            }
        }
        
        # 收集所有条目评分，用于计算总体平均值
        all_entry_scores = []
        total_facts = 0
        facts_passed_all = 0
        
        # 顺序处理每个条目
        for idx, entry in tqdm(enumerate(data), total=len(data), desc="处理条目"):
            print(f"\n处理条目 {idx + 1}/{len(data)}...")
            entry_result = self._process_entry(entry, idx)
            results["entry_results"].append(entry_result)
            results["entries_evaluated"] += 1
            
            # 收集条目评分
            entry_scores = entry_result["overall_scores"]
            all_entry_scores.append([
                entry_scores["average_accuracy"],
                entry_scores["average_self_containment"],
                entry_scores["average_format_compliance"],
                entry_scores["average_worthiness"]
            ])
            
            # 统计总事实数和全部通过的事实数
            for chunk_result in entry_result["chunk_results"]:
                for fact_eval in chunk_result["fact_evaluations"]:
                    total_facts += 1
                    if (fact_eval["accuracy_score"] == 1 and
                        fact_eval["self_containment_score"] == 1 and
                        fact_eval["format_compliance_score"] == 1 and
                        fact_eval["worthiness_score"] == 1):
                        facts_passed_all += 1
        
        # 计算总体统计
        if all_entry_scores:
            total_scores = [sum(category) for category in zip(*all_entry_scores)]
            avg_scores = [score / len(all_entry_scores) for score in total_scores]
            
            results["overall_statistics"] = {
                "average_accuracy": avg_scores[0],
                "average_self_containment": avg_scores[1],
                "average_format_compliance": avg_scores[2],
                "average_worthiness": avg_scores[3],
                "average_overall": sum(avg_scores) / 4,
                "total_facts_evaluated": total_facts,
                "facts_passed_all_criteria": facts_passed_all
            }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """生成评估报告"""
        report = []
        report.append("# 基于新EVAL_PROMPT的事实质量评估报告")
        report.append("")
        
        # 总体统计
        stats = results["overall_statistics"]
        report.append("## 总体统计")
        report.append(f"- 总条目数: {results['total_entries']}")
        report.append(f"- 已评估条目数: {results['entries_evaluated']}")
        report.append(f"- 总事实数: {stats['total_facts_evaluated']}")
        report.append(f"- 全部通过事实数: {stats['facts_passed_all_criteria']}")
        report.append(f"- 全部通过比例: {stats['facts_passed_all_criteria'] / stats['total_facts_evaluated']:.2%}  (假设总事实数>0)")
        report.append(f"- 平均准确性: {stats['average_accuracy']:.2%}")
        report.append(f"- 平均自包含性: {stats['average_self_containment']:.2%}")
        report.append(f"- 平均格式合规性: {stats['average_format_compliance']:.2%}")
        report.append(f"- 平均价值性: {stats['average_worthiness']:.2%}")
        report.append(f"- 平均总分: {stats['average_overall']:.2%}")
        report.append("")
        
        # 详细结果
        report.append("## 详细结果")
        for entry_result in results["entry_results"]:
            report.append(f"### 条目 {entry_result['entry_id']}")
            
            # 显示条目平均评分
            entry_scores = entry_result["overall_scores"]
            report.append(f"#### 条目平均评分")
            report.append(f"- 准确性: {entry_scores['average_accuracy']:.2%}")
            report.append(f"- 自包含性: {entry_scores['average_self_containment']:.2%}")
            report.append(f"- 格式合规性: {entry_scores['average_format_compliance']:.2%}")
            report.append(f"- 价值性: {entry_scores['average_worthiness']:.2%}")
            report.append(f"- 总分: {entry_scores['average_overall']:.2%}")
            report.append("")
            
            # 显示每个chunk的结果
            for chunk_result in entry_result["chunk_results"]:
                report.append(f"#### Chunk {chunk_result['chunk_idx']}")
                
                if not chunk_result["fact_evaluations"]:
                    report.append(f"- 此chunk没有提取到事实")
                    continue
                
                for fact_eval in chunk_result["fact_evaluations"]:
                    report.append(f"- **事实 {fact_eval['fact_idx']}**: {fact_eval['fact']['fact']}")
                    report.append(f"  - Details: {fact_eval['fact']['details']}")
                    report.append(f"  - 准确性: {'✅' if fact_eval['accuracy_score'] == 1 else '❌'} ({fact_eval['accuracy_score']})")
                    report.append(f"  - 自包含性: {'✅' if fact_eval['self_containment_score'] == 1 else '❌'} ({fact_eval['self_containment_score']})")
                    report.append(f"  - 格式合规性: {'✅' if fact_eval['format_compliance_score'] == 1 else '❌'} ({fact_eval['format_compliance_score']})")
                    report.append(f"  - 价值性: {'✅' if fact_eval['worthiness_score'] == 1 else '❌'} ({fact_eval['worthiness_score']})")
                    report.append(f"  - 评估理由: {fact_eval['reasoning']}")
                    report.append("")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, output_path: str):
        """保存评估结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def save_report(self, report: str, output_path: str):
        """保存评估报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基于新EVAL_PROMPT评估抽取事实质量')
    parser.add_argument('--input', type=str, default='/mnt/afs/codes/ljl/Memory-Agent/data/mem-alpha-facts-QA-4.1-1229.json', help='输入JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='输出结果目录')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='LLM模型名称')
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = NewFactQualityEvaluator(model_name=args.model)
    
    # 评估数据集
    print(f"开始评估数据集: {args.input}")
    results = evaluator.evaluate_dataset(args.input)
    
    # 生成报告
    report = evaluator.generate_report(results)
    
    # 保存结果
    import os
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 保存JSON结果
    json_output_path = os.path.join(args.output_dir, 'evaluation_results_new.json')
    evaluator.save_results(results, json_output_path)
    print(f"评估结果已保存到: {json_output_path}")
    
    # 保存报告
    report_output_path = os.path.join(args.output_dir, 'evaluation_report_new.md')
    evaluator.save_report(report, report_output_path)
    print(f"评估报告已保存到: {report_output_path}")
    
    # 打印总体统计
    print("\n总体评估结果:")
    stats = results['overall_statistics']
    print(f"- 平均准确性: {stats['average_accuracy']:.2%}")
    print(f"- 平均自包含性: {stats['average_self_containment']:.2%}")
    print(f"- 平均格式合规性: {stats['average_format_compliance']:.2%}")
    print(f"- 平均价值性: {stats['average_worthiness']:.2%}")
    print(f"- 平均总分: {stats['average_overall']:.2%}")
    print(f"- 全部通过事实数: {stats['facts_passed_all_criteria']} / {stats['total_facts_evaluated']}")
    if stats['total_facts_evaluated'] > 0:
        print(f"- 全部通过比例: {stats['facts_passed_all_criteria'] / stats['total_facts_evaluated']:.2%}")

if __name__ == "__main__":
    main()
