import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
import json

# this folder does not have model.safetensors
PATH = "./qwen3"
NUM_LAYERS = 12
HIDDEN_SIZE = 4096
FFN_SIZE = 12288
QH = 8
KH = 4
HEAD_DIM = 128
USE_NSA = True

NUM_LAYERS = 28
HIDDEN_SIZE = 2048
FFN_SIZE = 6144
QH = 16
KH = 8
HEAD_DIM = 128
USE_NSA = True

def replace_config():
    with open(os.path.join(PATH, "config.json"), 'r') as f:
        config = json.load(f)

    config["num_hidden_layers"] = NUM_LAYERS
    config["hidden_size"] = HIDDEN_SIZE
    config["head_dim"] = HEAD_DIM
    config["num_attention_heads"] = QH
    config["num_key_value_heads"] = KH
    config["intermediate_size"] = FFN_SIZE
    config["max_position_embeddings"] = 128 * 1024
    config.update(
        {
        "nsa": USE_NSA,
        "kernel_size": 32,
        "stride": 16,
        "block_size": 64,
        "top_n": 16,
        "num_init_blocks": 1,
        "num_local_blocks": 2,
        "window_size": 512,
        "fixed_y_maxlen": 512, # 512 * 16 = 8k
        "fixed_num_slc_blocks": 128, # 128 * 64 = 8k
        "num_cmp_splits": 0,
        "num_slc_splits": 0,
        }
    )

    with open(os.path.join(PATH, "config.json"), 'w') as f:
        f.write(json.dumps(config, ensure_ascii=False, indent=2))

def main():
    seed(0)

    num_seqs = 1024
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(4096, 8192-512))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(512, 2048)) for _ in range(num_seqs)]

    # don't exceed 8k.
    for i in range(num_seqs):
        sampling_params[i].max_tokens = min(8192 - len(prompt_token_ids[i]), sampling_params[i].max_tokens)

    replace_config()
    max_num_batched_tokens: int = 1024 * 128
    max_model_len: int = 1024 * 8
    gpu_memory_utilization: float = 0.95
    kvcache_block_size: int = 256 if not USE_NSA else 128

    llm = LLM(
        PATH, 
        enforce_eager=False, 
        max_model_len=max_model_len, 
        max_num_batched_tokens=max_num_batched_tokens, 
        gpu_memory_utilization=gpu_memory_utilization,
        kvcache_block_size=kvcache_block_size
    )

    t = time.time()
    _, time_dict = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False, return_time=True)
    t = time.time() - t

    prefill_time = sum(time_dict["prefill"])
    decode_time = sum(time_dict["decode"])
    prefill_tokens = sum([len(i) for i in prompt_token_ids]) 
    decode_tokens = sum([p.max_tokens for p in sampling_params])
    prefill_throughput = prefill_tokens / prefill_time
    decode_throughput = decode_tokens / decode_time

    print(f"nsa: {USE_NSA}")
    print(f"num_seqs: {num_seqs}")
    print(f"total_time: {t:.2f}s")
    print(f"total_prefill_tokens: {prefill_tokens}")
    print(f"total_decode_tokens: {decode_tokens}")
    print(f"prefill_throughput: {prefill_throughput:.2f}tok/s")
    print(f"decode_throughput: {decode_throughput:.2f}tok/s")

if __name__ == "__main__":
    main()
'''
nsa: False
num_seqs: 1024
total_time: 122.65s
total_prefill_tokens: 6050308
total_decode_tokens: 1209439
prefill_throughput: 104285.43tok/s
decode_throughput: 18742.70tok/s

nsa: True
num_seqs: 1024
total_time: 113.10s
total_prefill_tokens: 6050308
total_decode_tokens: 1209439
prefill_throughput: 90883.26tok/s
decode_throughput: 26049.62tok/s

1.7B

nsa: True
num_seqs: 1024
total_time: 211.62s
total_prefill_tokens: 6050308
total_decode_tokens: 1209439
prefill_throughput: 64788.45tok/s
decode_throughput: 10238.47tok/s

nsa: False
num_seqs: 1024
total_time: 268.07s
total_prefill_tokens: 6050308
total_decode_tokens: 1209439
prefill_throughput: 161714.76tok/s
decode_throughput: 5245.90tok/s
'''
