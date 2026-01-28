#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选longmemeval_s_cleaned.json数据脚本

功能：
- 根据指定的question_type筛选数据
- 支持多个question_type值
- 输出筛选前后的统计信息
- 将结果保存到新的JSON文件

使用示例：
python filter_data_by_type.py --input ./data/lme/longmemeval_s_cleaned.json --output ./filtered_data.json --types "single-session-user" "multi-session-user"
"""

import json
import argparse
from collections import Counter


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="筛选longmemeval_s_cleaned.json数据脚本")
    parser.add_argument("--input", type=str, default="./data/lme/longmemeval_s_cleaned.json", help="输入文件路径")
    parser.add_argument("--output", type=str, default="./filtered_data.json", help="输出文件路径")
    parser.add_argument("--types", type=str, nargs="+", required=True, help="要筛选的question_type列表")
    args = parser.parse_args()

    print(f"\n🔍 开始筛选数据...")
    print(f"   输入文件: {args.input}")
    print(f"   输出文件: {args.output}")
    print(f"   筛选类型: {args.types}")

    # 读取原始数据
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n📊 原始数据统计:")
        print(f"   总数据条数: {len(data)}")
        
        # 统计原始数据中各question_type的分布
        original_types = Counter(item.get("question_type", "unknown") for item in data)
        print(f"   各question_type分布:")
        for type_name, count in original_types.most_common():
            print(f"     {type_name}: {count}")
    except Exception as e:
        print(f"❌ 读取输入文件失败: {e}")
        return

    # 筛选符合条件的数据
    filtered_data = []
    for item in data:
        if item.get("question_type") in args.types:
            filtered_data.append(item)

    print(f"\n📊 筛选后数据统计:")
    print(f"   总数据条数: {len(filtered_data)}")
    
    # 统计筛选后数据中各question_type的分布
    filtered_types = Counter(item.get("question_type", "unknown") for item in filtered_data)
    print(f"   各question_type分布:")
    for type_name, count in filtered_types.most_common():
        print(f"     {type_name}: {count}")

    # 计算筛选比例
    if len(data) > 0:
        filter_ratio = len(filtered_data) / len(data) * 100
        print(f"   筛选比例: {filter_ratio:.2f}%")

    # 保存结果到新的JSON文件
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 筛选结果已保存到: {args.output}")
    except Exception as e:
        print(f"❌ 保存输出文件失败: {e}")
        return

    print(f"\n🎉 数据筛选完成!")


if __name__ == "__main__":
    main()
