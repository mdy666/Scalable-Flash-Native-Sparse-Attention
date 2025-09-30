import json
from dataclasses import dataclass, fields

def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n

@dataclass
class NSAConfig:
    hidden_size: int = 4096
    # base config
    head_dim: int = 128
    qk_head_dim: int = None
    v_head_dim: int = None
    num_heads: int = 64
    num_kv_heads: int = 4

    # nsa config
    kernel_size: int = 32
    stride: int = 16
    cmp_k_method: str = "mean"
    cmp_v_method: str = "mean"
    block_size: int = 64
    top_n: int = 16
    num_init_blocks: int = 1
    num_local_blocks: int = 2
    window_size: int = 512

    # recompute
    recompute_swa_o: bool = False
    recompute_slc_o: bool = False
    recompute_cmp_kv: bool = False
    recompute_cmp_o: bool = False

    # cp
    cp_mode: int = 1
    kv_head_stride: int = 1
    use_overlap_swa: bool = False

    def __post_init__(self):
        assert next_power_of_2(self.stride) == self.stride
        assert self.kernel_size % self.stride == 0
        assert self.block_size >= self.kernel_size
        assert self.block_size % self.stride == 0

        if self.qk_head_dim is None:
            assert self.head_dim is not None
            self.qk_head_dim = self.head_dim
        if self.v_head_dim is None:
            assert self.head_dim is not None
            self.v_head_dim = self.head_dim

        G = self.num_heads // self.num_kv_heads
        if G < 16 or next_power_of_2(G) != G:
            print(f"G = num_heads // num_kv_heads = {G}")
            print("For best performance, I suggest G>=16 and G is power of 2.")
            print("But don't worry about the correctness, any G is ok!")

    @classmethod
    def from_json(cls, json_file: str):

        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)


        field_defaults = {field.name: field.default for field in fields(cls)}
        print(field_defaults)
        for key in field_defaults:
            if key in json_data:
                field_defaults[key] = json_data[key]
        print(field_defaults["hidden_size"])
        instance = cls(**field_defaults)

        for key, value in json_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)

        return instance





