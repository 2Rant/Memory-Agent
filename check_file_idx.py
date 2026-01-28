import json

def check_file_match():
    file_path = "/mnt/afs/codes/ljl/Memory-Agent/data/lme/longmemeval_s_cleaned.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Check index 0
    idx = 0
    if idx < len(data):
        item = data[idx]
        print(f"File Index {idx}:")
        print(f"Question: {item.get('question')}")
        sessions = item.get("haystack_sessions", [])
        print(f"First Session Content: {sessions[0][0]['content'][:100] if sessions else 'N/A'}")
    else:
        print(f"Index {idx} out of range (len={len(data)})")

if __name__ == "__main__":
    check_file_match()
