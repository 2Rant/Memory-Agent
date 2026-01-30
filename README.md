# Memory-Agent
Recommend command:
python pipeline_chat_history_fact_mem_core.py --num_users -1 --max_workers 30 --retrieve_limit 10 --clear-db --extract-mode turn 2>&1 --data-path ./data/your/path --eval | tee pipeline_eval.log