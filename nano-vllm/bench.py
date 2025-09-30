import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams
import json
import pandas as pd

# this folder does not have model.safetensors
PATH = "./qwen3"
NUM_LAYERS = 12
HIDDEN_SIZE = 4096
FFN_SIZE = 12288
QH = 64
KH = 4
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
        "fixed_y_maxlen": 8192, # 8192 * 16 = 128k
        "fixed_num_slc_blocks": 2048, # 2048 * 64 = 128k
        "num_cmp_splits": 0,
        "num_slc_splits": 0,
        }
    )

    with open(os.path.join(PATH, "config.json"), 'w') as f:
        f.write(json.dumps(config, ensure_ascii=False, indent=2))

def main():
    replace_config()

    max_num_batched_tokens: int = 1024 * 128
    max_model_len: int = 1024 * 128
    gpu_memory_utilization: float = 0.95
    kvcache_block_size: int = 256 if not USE_NSA else 128

    # need change topk fixed shape if need bigger seqlen. (in + out <= 128k)
    SEQLENS = [1024 * i for i in [8, 16, 32, 64, 120]]
    # SEQLENS = [1024 * i for i in [8]]
    NUM_SEQS = [512, 512, 256, 256, 256]

    llm = LLM(
        PATH, 
        enforce_eager=False, 
        max_model_len=max_model_len, 
        max_num_batched_tokens=max_num_batched_tokens, 
        gpu_memory_utilization=gpu_memory_utilization,
        kvcache_block_size=kvcache_block_size
    )

    output_tokens = 256
    result = {"num_seqs":[], "seqlen": [], "output_tokens":[], "prefill_throughput":[], "decode_throughput":[]}
    for i in range(len(SEQLENS)):
        B = NUM_SEQS[i]
        S = SEQLENS[i]

        prompt_token_ids = [[randint(10000, 20000) for _ in range(S)] for _ in range(B)]
        sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens) for _ in range(B)]
        _, time_dict = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False, return_time=True)

        prefill_time = sum(time_dict["prefill"])
        decode_time = sum(time_dict["decode"])
        prefill_throughput = B * S / prefill_time
        decode_throughput = B * output_tokens / decode_time
        print(f"num_seqs: {B}, seqlen: {S}, output_tokens: {output_tokens}, prefill_throughput: {prefill_throughput:.2f}tok/s, decode_throughput: {decode_throughput:.2f}tok/s")

        result["num_seqs"].append(B)
        result["seqlen"].append(S)
        result["output_tokens"].append(output_tokens)
        result["prefill_throughput"].append(round(prefill_throughput, 2))
        result["decode_throughput"].append(round(decode_throughput, 2))

    df = pd.DataFrame(result)
    print(df)
    df.to_csv(f"./benchmark_result/qh={QH}-kh={KH}-head_dim={HEAD_DIM}-nsa={USE_NSA}.csv")

if __name__ == "__main__":
    main()
