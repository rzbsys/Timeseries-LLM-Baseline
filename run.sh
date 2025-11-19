vllm run-batch \
    -i cot_bts_biosignal_test_batch.jsonl \
    -o results_cot.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct
