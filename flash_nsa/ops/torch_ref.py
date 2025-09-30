import torch
import torch.nn.functional as F

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def torch_construct_block(x, kernel_size, stride):
    '''
    Args:
        x (Tensor): [bs, n, h, d]
        weight (Parameters): [kernel_size], 貌似可以加个维度h，作用看下面代码
        pe_embeding (Parameters): [kernel_size, d], 貌似也可以加个维度h，类似bert中的pe
        stride (int): 论文中的d
    Return:
        compress_x (Tensor): [bs, num_blocks, h, d]
    '''
    T, H, D = x.shape
    num_blocks = max((T - kernel_size) // stride + 1, 0)

    # [bs, h, num_blocks, kernel_size, D]
    block_x = torch.cat(
        [
        torch.roll(x, shifts=-1 * idx * stride, dims=0)[:num_blocks*stride].reshape(num_blocks, stride, H, D)
        for idx in range(kernel_size//stride)
        ], 
        axis=1
        )
    return block_x.transpose(1, 2)

def torch_swiglu(x):
    gate, up = x.chunk(2, -1)
    return F.silu(gate) * up

def torch_sigmoid_mul(x):
    gate, up = x.chunk(2, -1)
    return F.sigmoid(gate) * up


def torch_cmp_attn(q, k:torch.Tensor, v, kernel_size, stride):
    # q = q.unsqueeze(0).transpose(1, 2)
    # k = k.unsqueeze(0).transpose(1, 2)
    # v = v.unsqueeze(0).transpose(1, 2)
    # def score_mod(score, batch, head, q_idx, block_idx):
    #     k_idx = block_idx * stride + kernel_size - 1
    #     score = score + (q_idx < k_idx) * float('-inf')
    #     return score
    # out = flex_attention(q, k, v, enable_gqa=True, score_mod=score_mod)
    # return out.transpose(1,2).squeeze(0)

    k = k.repeat_interleave(q.size(1)//k.size(1), 1)
    v = v.repeat_interleave(q.size(1)//v.size(1), 1)
    sm_scale = q.size(-1) ** -0.5
    score = torch.einsum("nhd, mhd->hnm", q, k)
    q_idx = torch.arange(0, q.size(0), device=q.device, dtype=torch.int32)
    block_idx = torch.arange(0, k.size(0), device=q.device, dtype=torch.int32)
    k_idx = block_idx * stride + kernel_size - 1
    mask = q_idx[:, None] >= k_idx[None, :]
    score = torch.where(mask[None,:,:], score * sm_scale, float('-inf'))
    p = score.softmax(-1, dtype=torch.float32).to(q.dtype)
    p[:, :kernel_size-1] = 0
    out = torch.einsum("hnm,mhd->nhd", p, v)
    return out

def torch_topk(q, k:torch.Tensor, kernel_size, stride, block_size, topn, num_init, num_local, ignore_idx=99999999):
    n, qh, d = q.shape
    m, kh, d = k.shape
    G = qh // kh
    k = k.repeat_interleave(G, 1)
    sm_scale = q.size(-1) ** -0.5
    score = torch.einsum("nhd, mhd->hnm", q, k)
    q_idx = torch.arange(0, q.size(0), device=q.device, dtype=torch.int32)
    block_idx = torch.arange(0, k.size(0), device=q.device, dtype=torch.int32)
    k_idx = block_idx * stride + kernel_size - 1
    mask = q_idx[:, None] >= k_idx[None, :]
    score = torch.where(mask[None,:,:], score * sm_scale, float('-inf'))
    p = score.softmax(-1, dtype=torch.float32).to(q.dtype)
    p = p.view(kh, G, n, m).sum(1)
    p[:, :kernel_size-1] = 0

    num_slc_blocks = cdiv(n, block_size)
    topk = torch.full((kh, n, topn), ignore_idx, device=q.device, dtype=torch.int64)
    slc_p = torch.zeros(kh, block_size, num_slc_blocks, device=q.device, dtype=torch.float32)
    for start in range(0, n, block_size):
        num_loops = (block_size + kernel_size - stride) // stride
        cmp_start = -(kernel_size - stride)
        cmp_end = stride
        slc_start = 0
        slc_end = block_size
        cmp_idx = (torch.arange(0, num_slc_blocks, device=q.device, dtype=torch.int32) * block_size + cmp_start) // stride
        for _ in range(num_loops):
            if m == 0: 
                break
            area = (min(cmp_end, slc_end) - max(cmp_start, slc_start))*stride
            part_slc_p = p[:, start:start+block_size, cmp_idx%m]
            part_slc_p = torch.where(((cmp_idx>=0) & (cmp_idx<m))[None, None, :], part_slc_p * area, 0.)
            slc_p[:, :part_slc_p.size(1)] += part_slc_p
            cmp_idx += 1
            cmp_start += stride
            cmp_end += stride
        slc_p[:, :, :num_init] = 99999
        slc_p[:, :, max(cdiv(start, block_size)+1-num_local, 0): cdiv(start, block_size)+1] = 99999
        part_topk = torch.topk(slc_p[:, :min(block_size, n-start)], k=min(topn, start//block_size + 1), dim=-1)[1].to(torch.int32)
        topk[:, start:start+block_size, :part_topk.size(-1)] = part_topk
        slc_p.zero_()
    return topk


def torch_slc_attn(q, k, v, topk, block_size):
    n, qh, d = q.shape
    pad_n = cdiv(n, block_size) * block_size
    kh = k.size(1)
    sm_scale = q.size(-1) ** -0.5

    k = k.repeat_interleave(qh//kh, 1)
    v = v.repeat_interleave(qh//kh, 1)

    max_idx = topk[0, 0, -1]
    topk = topk.sort(-1)[0]
    topk = torch.where(topk == max_idx, 0, topk)
    indices = (topk[..., None] * block_size + torch.arange(0, block_size, device=q.device)[None, None, None, :]).flatten(-2, -1)
    slc_mask = torch.zeros(kh, n, pad_n, dtype=torch.bool, device=q.device)
    slc_mask.scatter_(-1, indices, 1)
    slc_mask = slc_mask[..., :n]
    idx = torch.arange(0, n, device=q.device, dtype=torch.int32)
    causal_mask = idx[:, None] >= idx[None, :]
    mask = causal_mask[None] & slc_mask

    score = torch.einsum("nhd, mhd->hnm", q, k)
    score = torch.where(mask.unsqueeze(1), score.view(kh, qh//kh, n ,n) * sm_scale, float('-inf')).flatten(0, 1)
    p = score.softmax(-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("hnm,mhd->nhd", p, v)
    return out

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def torch_reorder(x, world_size, shuffle_mode="shuffle"):
    assert len(x.shape) == 3
    assert x.is_contiguous()
    assert shuffle_mode in ["shuffle", "unshuffle"]
    T, H, D = x.shape
    assert T % (world_size * 2) == 0

    x_list = [t for t in x.chunk(2 * world_size)]
    if shuffle_mode == "shuffle":
        out_list = []
        for i in range(world_size):
            out_list.append(x_list[i])
            out_list.append(x_list[2 * world_size - 1 - i])
        return torch.cat(out_list, axis=0)
    else:
        return torch.cat(x_list[::2] + x_list[1::2][::-1], axis=0)


def torch_sigmoid_combine(a, b, c, w):
    return a * torch.nn.functional.sigmoid(w[..., 0].unsqueeze(-1)) + \
        b * torch.nn.functional.sigmoid(w[..., 1].unsqueeze(-1)) + \
        c * torch.nn.functional.sigmoid(w[..., 2].unsqueeze(-1))

