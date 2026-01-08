#!/usr/bin/env python3
"""
Convert parquet files to jsonl format.
Usage: python parquet_to_jsonl.py <input_parquet_files> [--output-dir <output_directory>]

Example:
python parquet_to_jsonl.py /path/to/file1.parquet /path/to/file2.parquet --output-dir /path/to/output
"""

import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

def parquet_to_jsonl(input_file, output_dir):
    """
    Convert a single parquet file to jsonl format with enhanced escape character handling.
    
    Args:
        input_file (str): Path to the input parquet file.
        output_dir (str): Directory to save the output jsonl file.
    """
    try:
        # Read parquet file using pandas
        df = pd.read_parquet(input_file)
        
        # Create output file name
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"memalpha-{os.path.splitext(base_name)[0]}.jsonl")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nConverting {len(df)} rows from {input_file} to {output_file}...")
        
        # Convert dataframe to jsonl with enhanced escape handling
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {base_name}"):
                # Convert row to dictionary, handling NaN values
                row_dict = row.dropna().to_dict()
                
                # Enhanced field fixing function
                def fix_field(value):
                    """Recursively fix all escape characters in any field"""
                    if isinstance(value, str):
                        # First try to parse as JSON directly
                        try:
                            parsed = json.loads(value)
                            # If successful, recursively fix the parsed value
                            if isinstance(parsed, dict):
                                return {k: fix_field(v) for k, v in parsed.items()}
                            elif isinstance(parsed, list):
                                return [fix_field(item) for item in parsed]
                            return parsed
                        except:
                            # If direct parsing fails, try with cleaned quotes
                            # This handles cases where string has extra surrounding quotes
                            stripped_value = value.strip('"')
                            if stripped_value != value:
                                try:
                                    parsed = json.loads(stripped_value)
                                    if isinstance(parsed, dict):
                                        return {k: fix_field(v) for k, v in parsed.items()}
                                    elif isinstance(parsed, list):
                                        return [fix_field(item) for item in parsed]
                                    return parsed
                                except:
                                    pass
                            
                            # If still not JSON, clean up escape sequences for regular string
                            fixed_value = value
                            # Fix escaped quotes first to prevent double replacement
                            fixed_value = fixed_value.replace('\\"', '"')
                            # Fix double backslashes
                            fixed_value = fixed_value.replace('\\\\', '\\')
                            # Fix escaped control characters
                            fixed_value = fixed_value.replace('\\n', '\n')
                            fixed_value = fixed_value.replace('\\t', '\t')
                            fixed_value = fixed_value.replace('\\r', '\r')
                            fixed_value = fixed_value.replace('\\b', '\b')
                            fixed_value = fixed_value.replace('\\f', '\f')
                            return fixed_value
                    elif isinstance(value, dict):
                        # Recursively fix dictionary fields
                        return {k: fix_field(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        # Recursively fix list items
                        return [fix_field(item) for item in value]
                    return value
                
                # Fix all fields in the row_dict
                for key, value in list(row_dict.items()):
                    row_dict[key] = fix_field(value)
                
                # Ensure chunks is always a list
                if 'chunks' in row_dict:
                    chunks = row_dict['chunks']
                    if not isinstance(chunks, list):
                        row_dict['chunks'] = [chunks] if chunks else []
                
                # Special handling for common problematic fields
                for field in ['sub_source', 'metadata', 'questions_and_answers', 'prompt']:
                    if field in row_dict:
                        value = row_dict[field]
                        if isinstance(value, str):
                            # Final cleanup for problematic fields
                            value = value.replace('\\n', '\n')
                            value = value.replace('\\t', '\t')
                            value = value.replace('\\"', '"')
                            value = value.replace('\\\\', '\\')
                            row_dict[field] = value
                
                # Write as JSON line with proper formatting
                json.dump(row_dict, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"\n✓ Successfully converted {input_file} to {output_file}")
        print(f"  - Total rows processed: {len(df)}")
        print(f"  - Columns found: {list(df.columns)}")
        
    except Exception as e:
        print(f"\n✗ Error converting {input_file}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Convert parquet files to jsonl format")
    parser.add_argument('input_files', nargs='+', help='Input parquet files')
    parser.add_argument('--output-dir', default='/mnt/afs/codes/ljl/Memory-Agent/data', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Converting {len(args.input_files)} parquet files to jsonl...")
    print(f"Output directory: {args.output_dir}")
    print()
    
    for input_file in args.input_files:
        if os.path.exists(input_file):
            parquet_to_jsonl(input_file, args.output_dir)
        else:
            print(f"✗ File not found: {input_file}")
        print()
    
    print("Conversion completed!")

if __name__ == "__main__":
    main()
