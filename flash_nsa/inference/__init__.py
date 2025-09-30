from .combine import combine_decode, combine_prefill
from .topk import topk_decode, topk_prefill
from .compress_kv import mean_pooling_decode, mean_pooling_prefill, kv_mean_pooling_decode
from .compress_attn import cmp_attn_decode, cmp_attn_prefill
from .select_attn import slc_attn_decode, fused_slc_swa_attn_decode, slc_attn_prefill
from flash_attn_interface import flash_attn_with_kvcache

def nsa_prefill(
    q, 
    k, 
    v, 
    w, 
    x_cu_seqlens,
    y_cu_seqlens,
    x_maxlen,
    y_maxlen,
    context_lens,
    block_tables,
    cmp_slot_mapping,
    cmp_block_tables,
    fixed_y_maxlen = 8192,
    fixed_num_slc_blocks = 2048,
    kernel_size = 32,
    stride = 16,
    block_size = 64,
    top_n = 16,
    num_init_blocks = 1,
    num_local_blocks = 2,
    window_size = 512,
    sm_scale = None,
    ):
    '''
    NSA prefill function, now only support kv meanpooling

    Args:
        q (torch.Tensor): [t, qh, d], t = sum(context_lens) - total_num_cacahe_tokens
        k (torch.Tensor): [total_blocks, page_size, kh, d]
        v (torch.Tensor): [total_blocks, page_size, kh, vd]
        w (torch.Tensor): [t, qh, 3]
        x_cu_seqlens (torch.Tensor): [bs + 1], cu_seqlens_q
        y_cu_seqlens (torch.Tensor): [bs + 1], cu_seqlens_cmp_kv
        x_maxlen (int): max_len_q
        y_maxlen (int): max_len_cmp_kv
        context_lens (torch.Tensor): [bs], True length, contain cached tokens
        block_tabels (torch.Tensor): [bs, N1], kv mapping
        cmp_slot_mapping (torch.Tensor): [y_cu_seqlens[-1].item()], we dont use share cmp kv cache
        cmp_block_tabels (torch.Tensor): [bs, N2], cmp_kv mapping
        fixed_y_maxlen (int): For topk, fixed the args and tensor shape. It's useful when enable cuda graph. 8192 for (max_len<=128k)
        num_slc_blocks (int): For topk, fixed the BLOCK_K: tl.contextper. It's useful when enable cuda graph. 2048 for (max_len<=128k)
        ......
        NSA normal args
        ......

    Return:
        o (torch.Tensor): [t, qh, vd]

    '''
    assert k.size(1) % 128 == 0
    
    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5

    mean_pooling_prefill(k, y_cu_seqlens, y_maxlen, block_table=block_tables, slot_mapping=cmp_slot_mapping, kernel_size=kernel_size, stride=stride)
    mean_pooling_prefill(v, y_cu_seqlens, y_maxlen, block_table=block_tables, slot_mapping=cmp_slot_mapping, kernel_size=kernel_size, stride=stride)

    cmp_o, lse = cmp_attn_prefill(
        q,
        k,
        v,
        x_cu_seqlens,
        x_maxlen,
        block_tables=cmp_block_tables,
        context_lens=context_lens,
        kernel_size=kernel_size,
        stride=stride,
        sm_scale=sm_scale
        )

    topk = topk_prefill(
        q,
        k,
        lse,
        x_cu_seqlens,
        x_maxlen,
        y_maxlen,
        block_tables=cmp_block_tables,
        context_lens=context_lens,
        fixed_y_maxlen=fixed_y_maxlen,
        fixed_num_slc_blocks=fixed_num_slc_blocks,
        sm_scale=sm_scale,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        top_n=top_n,
        num_inital=num_init_blocks,
        num_local=num_local_blocks
        )

    slc_o, _ = slc_attn_prefill(
        q,
        k,
        v,
        topk,
        x_cu_seqlens,
        x_maxlen,
        block_tables=block_tables,
        context_lens=context_lens, 
        block_size=block_size,
        top_n=top_n,
        sm_scale=sm_scale
        )

    swa_o = flash_attn_with_kvcache(
        q,
        k,
        v,
        cu_seqlens_q=x_cu_seqlens,
        max_seqlen_q=x_maxlen,
        page_table=block_tables, 
        cache_seqlens=context_lens,
        window_size=(window_size, -1),
        causal=True,
        softmax_scale=sm_scale
        )

    o = combine_prefill(cmp_o, slc_o, swa_o, w)
    return o

def nsa_decode(
    q, 
    k, 
    v, 
    w, 
    context_lens,
    block_tables,
    cmp_slot_mapping,
    cmp_block_tables,
    cmp_num_splits=0,
    slc_num_splits=0,
    fixed_y_maxlen=8192,
    fixed_num_slc_blocks=2048,
    kernel_size = 32,
    stride = 16,
    block_size = 64,
    top_n = 16,
    num_init_blocks = 1,
    num_local_blocks = 2,
    window_size = 512,
    sm_scale = None,
    ):
    '''
    NSA decode function, now only support kv meanpooling

    Args:
        q (torch.Tensor): [b, qh, d]
        k (torch.Tensor): [total_blocks, page_size, kh, d]
        v (torch.Tensor): [total_blocks, page_size, kh, vd]
        w (torch.Tensor): [b, qh, 3]
        context_lens (torch.Tensor): [bs], True length, contain cached tokens
        block_tabels (torch.Tensor): [bs, N1], kv mapping
        cmp_slot_mapping (torch.Tensor): [bs], -1 means don't have a new cmp_kv
        cmp_block_tabels (torch.Tensor): [bs, N2], cmp_kv mapping
        cmp_num_splits (int): Maybe can be a tensor [BS] like sglang
        slc_num_splits (int): top_n and window_size is fixed, so a int num is ok
        fixed_y_maxlen (int): For topk, fixed the args and tensor shape. It's useful when enable cuda graph. 8192 for (max_len<=128k)
        num_slc_blocks (int): For topk, fixed the BLOCK_K: tl.contextper. It's useful when enable cuda graph. 2048 for (max_len<=128k)
        ......
        NSA normal args
        ......

    Return:
        o (torch.Tensor): [t, qh, vd]

    '''
    assert k.size(1) % 128 == 0

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5

    kv_mean_pooling_decode(k, v, block_table=block_tables, slot_mapping=cmp_slot_mapping, context_lens=context_lens, kernel_size=kernel_size, stride=stride)

    cmp_o, lse = cmp_attn_decode(
        q,
        k,
        v,
        block_tables=cmp_block_tables,
        context_lens=context_lens,
        num_splits=cmp_num_splits,
        kernel_size=kernel_size,
        stride=stride,
        sm_scale=sm_scale
        )

    topk = topk_decode(
        q,
        k,
        lse,
        fixed_y_maxlen=fixed_y_maxlen,
        fixed_num_slc_blocks=fixed_num_slc_blocks,
        block_tables=cmp_block_tables,
        context_lens=context_lens,
        kernel_size=kernel_size,
        stride=stride,
        block_size=block_size,
        top_n=top_n,
        num_inital=num_init_blocks,
        num_local=num_local_blocks,
        sm_scale=sm_scale,
        )

    slc_o, swa_o = fused_slc_swa_attn_decode(
        q,
        k,
        v,
        topk,
        num_splits=slc_num_splits,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
        top_n=top_n,
        window_size=window_size,
        sm_scale=sm_scale
        )

    o = combine_decode(cmp_o, slc_o, swa_o, w)
    return o