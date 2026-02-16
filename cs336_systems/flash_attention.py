import math
import torch
from einops import einsum
import triton
import triton.language as tl



class FlashAttn2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        # Ignore causal for part (a) as required
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        assert D == Dk
        assert V.shape[:2] == (B, Nk)

        # choose tile sizes (>=16)
        Bq = 16
        Bk = 16

        scale = 1.0 / math.sqrt(D)

        # output buffers
        O = torch.empty((B, Nq, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)  # usually fp32

        # tiled loops
        for b in range(B):
            for qi in range(0, Nq, Bq):
                q = Q[b, qi:qi+Bq, :] # shape: (Bq, D)
                # fp32 accumulators
                m = torch.full((q.shape[0],), -float("inf"), device=Q.device, dtype=torch.float32)
                l = torch.zeros((q.shape[0],), device=Q.device, dtype=torch.float32)
                o = torch.zeros((q.shape[0], D), device=Q.device, dtype=torch.float32)

                q = q.to(torch.float32)

                for kj in range(0, Nk, Bk):
                    k = K[b, kj:kj+Bk, :].to(torch.float32)  # shape: (Bk, D)
                    v = V[b, kj:kj+Bk, :]                    # shape: (Bk, D), dtype: V.dtypedtype

                    # S[q, k] = sum_d q[q, d] * k[k, d]
                    s = einsum(q, k, 'q d, k d -> q k') * scale  # (Bq, Bk) fp32

                    m_new = torch.maximum(m, s.max(dim=1).values)   # (Bq,)
                    p = torch.exp(s - m_new[:, None])               # (Bq, Bk) fp32

                    # 前 kj - 1 个 tile 的 loss 累加为 m * l
                    # 第 kj 个 tile 的 loss 为 m_new * p.sum(dim=1)
                    # 统一写成 exp(m_new) * l_new
                    l_new = torch.exp(m - m_new) * l + p.sum(dim=1) # (Bq,)

                    # (P @ V)[q, d] = sum_k p[q, k] * v[k, d]
                    pv = einsum(p.to(v.dtype), v, 'q k, k d -> q d').to(torch.float32)
                    o = o * torch.exp(m - m_new)[:, None] + pv
                    m, l = m_new, l_new

                o = o / l[:, None]
                O[b, qi:qi+q.shape[0], :] = o.to(Q.dtype)
                L[b, qi:qi+q.shape[0]] = m + torch.log(l)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError



@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Q block (Bq, D)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Output blocks
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load Q once
    q = tl.load(Q_block_ptr).to(tl.float32)  # (Bq, D)

    # On-chip accumulators
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    # indices for causal mask (optional)
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    # Loop over key tiles
    for kj in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(kj * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(kj * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr).to(tl.float32)  # (Bk, D)
        v = tl.load(V_block_ptr)                 # (Bk, D) keep element type

        # S = QK^T * scale  => (Bq, Bk)
        s = tl.dot(q, tl.trans(k)) * scale

        if is_causal:
            k_idx = kj * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]           # True means future (mask out)
            s = s + tl.where(mask, -1e6, 0.0).to(tl.float32)

        # online softmax update
        rowmax = tl.max(s, axis=1)
        m_new = tl.maximum(m, rowmax)
        p = tl.exp(s - m_new[:, None])                       # (Bq,Bk) fp32
        l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)    # (Bq,)

        # O update
        alpha = tl.exp(m - m_new)                            # (Bq,)
        o = o * alpha[:, None]
        # cast p to V dtype before dot, but accumulate into fp32 o via acc
        p_cast = p.to(V_block_ptr.type.element_ty)
        o = tl.dot(p_cast, v, acc=o).to(tl.float32)

        m, l = m_new, l_new

    # finalize
    o = o / l[:, None]
    L = m + tl.log(l)

    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L)


class FlashAttn2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        assert D == Dk and V.shape[:2] == (B, Nk)

        # tile sizes (>=16)
        Q_TILE = 16
        K_TILE = 16
        scale = 1.0 / math.sqrt(D)

        O = torch.empty((B, Nq, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        grid = (Nq // Q_TILE, B)  # dimensions are clean powers of 2 and >=16 per spec

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=Nq, N_KEYS=Nk,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE,
            K_TILE_SIZE=K_TILE,
            is_causal=is_causal,
            num_warps=4,  # ok default; can tune later
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError


