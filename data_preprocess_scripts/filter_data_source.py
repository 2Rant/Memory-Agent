#!/usr/bin/env python3
"""
Filter JSONL file to keep only rows with specific data sources.
Usage: python filter_data_source.py <input_file> <output_file> [--sources SOURCE1 SOURCE2 ...]

Example:
python filter_data_source.py /mnt/afs/codes/ljl/Memory-Agent/data/memalpha-train.jsonl /mnt/afs/codes/ljl/Memory-Agent/data/memalpha-train-filtered.jsonl --sources hotpotqa squad lme_train perltqa
"""

import json
import argparse
from tqdm import tqdm

def filter_jsonl_by_data_source(input_file, output_file, target_sources):
    """
    Filter JSONL file to keep only rows with specified data sources.
    
    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output filtered JSONL file.
        target_sources (list): List of data sources to keep.
    """
    try:
        # Count total lines for progress bar
        total_lines = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
        
        kept_count = 0
        skipped_count = 0
        
        # Filter the file
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, total=total_lines, desc=f"Processing {input_file}"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    data_source = data.get('data_source', '')
                    
                    if data_source in target_sources:
                        # Write the line as-is to preserve formatting
                        f_out.write(line)
                        f_out.write('\n')
                        kept_count += 1
                    else:
                        skipped_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    skipped_count += 1
        
        print(f"\n✓ Filtering completed!")
        print(f"  - Input file: {input_file}")
        print(f"  - Output file: {output_file}")
        print(f"  - Total lines processed: {total_lines}")
        print(f"  - Lines kept: {kept_count}")
        print(f"  - Lines skipped: {skipped_count}")
        print(f"  - Target data sources: {', '.join(target_sources)}")
        
    except Exception as e:
        print(f"\n✗ Error filtering {input_file}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Filter JSONL file by data source")
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('output_file', help='Output filtered JSONL file')
    parser.add_argument('--sources', nargs='+', 
                      default=['hotpotqa', 'squad', 'lme_train', 'perltqa'],
                      help='Data sources to keep (default: hotpotqa squad lme_train perltqa)')
    
    args = parser.parse_args()
    
    filter_jsonl_by_data_source(args.input_file, args.output_file, args.sources)

if __name__ == "__main__":
    main()
