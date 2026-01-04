#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO Memory Trainer - 将Memory-Agent与GRPO算法集成的训练脚本

GRPO (Gradient-free Policy Optimization) 是一种强化学习算法，用于优化记忆操作策略
该脚本将Memory-Agent的记忆管理功能与GRPO训练相结合，实现智能记忆操作
"""

import os
import json
import numpy as np
import random
from copy import deepcopy
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

# 导入Memory-Agent组件
from pipeline_core_mem import (
    init_milvus_connection,
    MemReaderAgent,
    MemoryManager,
    process_user_memory
)

# GRPO算法参数
GRPO_CONFIG = {
    "num_epochs": 10,
    "batch_size": 10,
    "exploration_rate": 0.2,
    "learning_rate": 0.1,
    "discount_factor": 0.95
}

class GRPOMemoryTrainer:
    """GRPO记忆训练器，用于优化记忆操作策略"""
    
    def __init__(self, milvus_collection, llm_client):
        self.milvus_collection = milvus_collection
        self.llm_client = llm_client
        self.mem_reader_agent = MemReaderAgent(llm_client)
        self.memory_manager = MemoryManager(milvus_collection, llm_client)
        
        # GRPO策略参数
        self.policy_params = self._initialize_policy_params()
        
    def _initialize_policy_params(self):
        """初始化GRPO策略参数"""
        return {
            "add_weight": 0.25,
            "update_weight": 0.25,
            "delete_weight": 0.25,
            "none_weight": 0.25
        }
    
    def calculate_reward(self, action_type, input_text, old_memory, new_memory, all_memories):
        """计算记忆操作的奖励"""
        # 基于toy_reward.py的奖励机制
        
        # 1. ADD操作
        if action_type == 'ADD':
            # 计算新记忆与库中现有记忆的最大相似度
            max_sim = self._get_max_similarity(new_memory, all_memories)
            if max_sim > 0.85:
                return -1.0  # 惩罚：太像旧记忆
            else:
                return +1.0  # 奖励：真正的新信息
        
        # 2. UPDATE操作
        elif action_type == 'UPDATE':
            # 计算新旧记忆的相似度
            similarity = self._get_similarity(old_memory, new_memory)
            info_gain = self._get_information_gain(old_memory, new_memory)
            
            if similarity > 0.95:
                return -0.5  # 惩罚：内容几乎没变
            elif similarity < 0.2:
                return -1.0  # 惩罚：完全不相关
            else:
                return +1.0 + info_gain  # 奖励：有效更新
        
        # 3. DELETE操作
        elif action_type == 'DELETE':
            # 检查是否真的需要删除
            is_contradiction = self._check_nli_contradiction(input_text, old_memory)
            if is_contradiction:
                return +2.0  # 大奖：正确删除
            else:
                return -2.0  # 重罚：乱删记忆
        
        # 4. NONE操作
        elif action_type == 'NONE':
            # 检查是否包含新信息
            has_new_info = self._check_if_contains_new_info(input_text, all_memories)
            if has_new_info:
                return -1.0  # 惩罚：漏掉信息
            else:
                return +0.5  # 奖励：正确忽略
        
        return 0.0
    
    def _get_max_similarity(self, new_memory, all_memories):
        """获取新记忆与现有记忆的最大相似度"""
        if not all_memories:
            return 0.0
        # 简化实现，实际应使用向量相似度
        return max(0.0, 0.5)  # 占位实现
    
    def _get_similarity(self, old_memory, new_memory):
        """获取新旧记忆的相似度"""
        # 简化实现
        return 0.5
    
    def _get_information_gain(self, old_memory, new_memory):
        """计算信息增益"""
        # 简化实现
        return 0.1
    
    def _check_nli_contradiction(self, input_text, old_memory):
        """检查输入是否与旧记忆矛盾"""
        # 简化实现
        return False
    
    def _check_if_contains_new_info(self, input_text, all_memories):
        """检查输入是否包含新信息"""
        # 简化实现
        return True
    
    def grpo_step(self, state, action, reward, next_state):
        """GRPO算法的单步更新"""
        # GRPO核心更新逻辑
        # 简化实现：基于奖励调整策略参数
        if action == 'ADD':
            self.policy_params['add_weight'] += GRPO_CONFIG['learning_rate'] * reward
        elif action == 'UPDATE':
            self.policy_params['update_weight'] += GRPO_CONFIG['learning_rate'] * reward
        elif action == 'DELETE':
            self.policy_params['delete_weight'] += GRPO_CONFIG['learning_rate'] * reward
        elif action == 'NONE':
            self.policy_params['none_weight'] += GRPO_CONFIG['learning_rate'] * reward
        
        # 归一化参数
        total = sum(self.policy_params.values())
        for key in self.policy_params:
            self.policy_params[key] /= total
    
    def select_action(self, state):
        """基于当前状态选择动作"""
        # epsilon-greedy探索策略
        if random.random() < GRPO_CONFIG['exploration_rate']:
            # 随机选择动作
            return random.choice(['ADD', 'UPDATE', 'DELETE', 'NONE'])
        else:
            # 基于策略参数选择动作
            actions = list(self.policy_params.keys())
            weights = list(self.policy_params.values())
            return random.choices(actions, weights=weights, k=1)[0]
    
    def train(self, training_data):
        """训练主循环"""
        print(f"开始GRPO训练，共{GRPO_CONFIG['num_epochs']}轮，每轮{GRPO_CONFIG['batch_size']}个样本")
        
        for epoch in range(GRPO_CONFIG['num_epochs']):
            epoch_reward = 0.0
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), GRPO_CONFIG['batch_size']):
                batch = training_data[i:i+GRPO_CONFIG['batch_size']]
                
                for line in batch:
                    # 1. 获取当前状态
                    current_state = {
                        'dialogue': line['query'],
                        'existing_memories': self.memory_manager.search_memory(line['query']),
                        'core_memory': self.mem_reader_agent.core_memory
                    }
                    
                    # 2. 选择并执行动作
                    action = self.select_action(current_state)
                    
                    # 3. 执行记忆操作（使用Memory-Agent）
                    result = process_user_memory(line)
                    actual_action = result['memory_count']
                    
                    # 4. 获取下一个状态
                    next_state = {
                        'dialogue': line['query'],
                        'existing_memories': self.memory_manager.search_memory(line['query']),
                        'core_memory': result['core_memory']
                    }
                    
                    # 5. 计算奖励
                    reward = self.calculate_reward(
                        action_type=action,
                        input_text=line['query'],
                        old_memory=None,
                        new_memory=line['query'],
                        all_memories=next_state['existing_memories']
                    )
                    
                    # 6. GRPO更新
                    self.grpo_step(current_state, action, reward, next_state)
                    
                    epoch_reward += reward
            
            print(f"第{epoch+1}轮训练完成，平均奖励: {epoch_reward/len(training_data):.4f}")
            print(f"当前策略参数: {self.policy_params}")
    
    def evaluate(self, test_data):
        """评估训练后的策略"""
        print("\n开始评估训练后的策略...")
        total_reward = 0.0
        
        for line in test_data:
            # 1. 获取当前状态
            current_state = {
                'dialogue': line['query'],
                'existing_memories': self.memory_manager.search_memory(line['query']),
                'core_memory': self.mem_reader_agent.core_memory
            }
            
            # 2. 选择动作（无探索）
            original_exploration = GRPO_CONFIG['exploration_rate']
            GRPO_CONFIG['exploration_rate'] = 0.0
            action = self.select_action(current_state)
            GRPO_CONFIG['exploration_rate'] = original_exploration
            
            # 3. 执行记忆操作
            result = process_user_memory(line)
            
            # 4. 计算奖励
            reward = self.calculate_reward(
                action_type=action,
                input_text=line['query'],
                old_memory=None,
                new_memory=line['query'],
                all_memories=result['existing_memories'] if 'existing_memories' in result else []
            )
            
            total_reward += reward
        
        avg_reward = total_reward / len(test_data)
        print(f"评估完成，平均奖励: {avg_reward:.4f}")
        return avg_reward

def main():
    """主函数"""
    try:
        # 初始化Milvus连接
        milvus_collection = init_milvus_connection()
        print("成功初始化Milvus连接")
        
        # 创建GRPO训练器
        trainer = GRPOMemoryTrainer(milvus_collection, openai_client)
        
        # 加载训练数据
        try:
            with open("./data/longmemeval_s_cleaned.json", "r") as f:
                all_data = json.load(f)
            
            # 分割训练集和测试集
            split_index = int(len(all_data) * 0.8)
            training_data = all_data[:split_index]
            test_data = all_data[split_index:]
            
            print(f"加载了{len(all_data)}个样本，训练集{len(training_data)}个，测试集{len(test_data)}个")
            
            # 开始训练
            trainer.train(training_data)
            
            # 评估模型
            trainer.evaluate(test_data)
            
        except FileNotFoundError:
            print("警告：未找到训练数据文件，使用示例数据")
            # 使用示例数据
            sample_data = [
                {"query": "我喜欢苹果", "answer": "好的，我记住了您喜欢苹果"},
                {"query": "我现在也喜欢香蕉了", "answer": "好的，我更新了您的偏好"},
                {"query": "其实我不喜欢苹果了", "answer": "好的，我删除了您喜欢苹果的记录"},
                {"query": "今天天气不错", "answer": "是的，今天天气很好"}
            ]
            trainer.train(sample_data)
            trainer.evaluate(sample_data)
            
    except Exception as e:
        print(f"训练过程中发生错误: {e}")

if __name__ == "__main__":
    main()
