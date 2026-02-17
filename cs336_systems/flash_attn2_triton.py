import math
import torch
import triton
import triton.language as tl



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



@triton.jit
def compute_D_kernel(
    dO_ptr, O_ptr, D_ptr,
    stride_dob, stride_doq, stride_dod,
    stride_ob,  stride_oq,  stride_od,
    stride_db, stride_dq,
    NQ, D: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    # grid = (Tq, B)
    q_tile = tl.program_id(0)
    b = tl.program_id(1)

    # offsets for query rows in this tile
    q_idx = q_tile * Q_TILE + tl.arange(0, Q_TILE)  # (Q_TILE,)
    mask_q = q_idx < NQ

    # build pointers for (Q_TILE, D)
    d = tl.arange(0, D)  # assume D is power of 2 per spec; else you'd tile D too
    # (Q_TILE, D) pointers
    dO = tl.load(
        dO_ptr + b * stride_dob + q_idx[:, None] * stride_doq + d[None, :] * stride_dod,
        mask=mask_q[:, None],
        other=0.0,
    ).to(tl.float32)
    O = tl.load(
        O_ptr + b * stride_ob + q_idx[:, None] * stride_oq + d[None, :] * stride_od,
        mask=mask_q[:, None],
        other=0.0,
    ).to(tl.float32)

    # rowsum over D
    D_row = tl.sum(dO * O, axis=1)  # (Q_TILE,)

    tl.store(
        D_ptr + b * stride_db + q_idx * stride_dq,
        D_row,
        mask=mask_q
    )



@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, dO_ptr,
    L_ptr, D_ptr,        # L = logsumexp, D = rowsum(dO ∘ O)
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    NQ, NK,
    scale,
    D: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # grid = (Tq, Tk, B)
    q_tile = tl.program_id(0)
    k_tile = tl.program_id(1)
    b = tl.program_id(2)

    q_idx = q_tile * Q_TILE + tl.arange(0, Q_TILE)  # (Q_TILE,)
    k_idx = k_tile * K_TILE + tl.arange(0, K_TILE)  # (K_TILE,)
    mask_q = q_idx < NQ
    mask_k = k_idx < NK

    d = tl.arange(0, D)

    # Load Q tile: (Q_TILE, D)
    Q = tl.load(
        Q_ptr + b * stride_qb + q_idx[:, None] * stride_qq + d[None, :] * stride_qd,
        mask=mask_q[:, None],
        other=0.0,
    ).to(tl.float32)

    # Load dO tile: (Q_TILE, D)
    dO = tl.load(
        dO_ptr + b * stride_dob + q_idx[:, None] * stride_doq + d[None, :] * stride_dod,
        mask=mask_q[:, None],
        other=0.0,
    ).to(tl.float32)

    # Load L and D vectors: (Q_TILE,)
    L = tl.load(L_ptr + b * stride_lb + q_idx * stride_lq, mask=mask_q, other=-float("inf")).to(tl.float32)
    Drow = tl.load(D_ptr + b * stride_db + q_idx * stride_dq, mask=mask_q, other=0.0).to(tl.float32)

    # Load K, V tiles: (K_TILE, D)
    K = tl.load(
        K_ptr + b * stride_kb + k_idx[:, None] * stride_kk + d[None, :] * stride_kd,
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)

    V = tl.load(
        V_ptr + b * stride_vb + k_idx[:, None] * stride_vk + d[None, :] * stride_vd,
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)

    S = tl.dot(Q, tl.trans(K)) * scale #  (Q_TILE, K_TILE)

    if is_causal:
        # mask out future keys: if k > q then add -1e6
        causal = q_idx[:, None] < k_idx[None, :]
        S = S + tl.where(causal, -1e6, 0.0).to(tl.float32)

    P = tl.exp(S - L[:, None])  # (Q_TILE, K_TILE) fp32

    dV_local = tl.dot(tl.trans(P), dO)  # (K_TILE, D) fp32
    dP = tl.dot(dO, tl.trans(V))  # (Q_TILE, K_TILE) fp32
    dS = P * (dP - Drow[:, None])

    dQ_local = tl.dot(dS, K) * scale # (Q_TILE, D)
    dK_local = tl.dot(tl.trans(dS), Q) * scale # (K_TILE, D)

    # dQ: (Q_TILE, D)
    tl.atomic_add(
        dQ_ptr + b * stride_dqb + q_idx[:, None] * stride_dqq + d[None, :] * stride_dqd,
        dQ_local,
        mask=mask_q[:, None],
    )

    # dK: (K_TILE, D)
    tl.atomic_add(
        dK_ptr + b * stride_dkb + k_idx[:, None] * stride_dkk + d[None, :] * stride_dkd,
        dK_local,
        mask=mask_k[:, None],
    )

    # dV: (K_TILE, D)
    tl.atomic_add(
        dV_ptr + b * stride_dvb + k_idx[:, None] * stride_dvk + d[None, :] * stride_dvd,
        dV_local,
        mask=mask_k[:, None],
    )



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
    def backward(ctx, dO):
        # saved: Q,K,V,O,L  (L must be logsumexp per row)
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = getattr(ctx, "is_causal", False)

        B, NQ, D = Q.shape
        _, NK, _ = K.shape
        scale = 1.0 / math.sqrt(D)

        # tile sizes (>=16); assume powers of 2 for tests
        Q_TILE = 64
        K_TILE = 64
        assert NQ % Q_TILE == 0 and NK % K_TILE == 0 and D >= 16

        # allocate Dvec (fp32)
        Dvec = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        # grads in fp32 (atomic accumulation safer)
        dQ = torch.zeros((B, NQ, D), device=Q.device, dtype=torch.float32)
        dK = torch.zeros((B, NK, D), device=Q.device, dtype=torch.float32)
        dV = torch.zeros((B, NK, D), device=Q.device, dtype=torch.float32)

        # 1) compute D = rowsum(dO ∘ O)
        grid_D = (NQ // Q_TILE, B)
        compute_D_kernel[grid_D](
            dO, O, Dvec,
            dO.stride(0), dO.stride(1), dO.stride(2),
            O.stride(0),  O.stride(1),  O.stride(2),
            Dvec.stride(0), Dvec.stride(1),
            NQ, D=D,
            Q_TILE=Q_TILE,
            num_warps=4,
        )

        # 2) backward tiles (Tq, Tk, B)
        grid_bwd = (NQ // Q_TILE, NK // K_TILE, B)
        flash_bwd_kernel[grid_bwd](
            Q, K, V,
            O, dO,
            L, Dvec,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Dvec.stride(0), Dvec.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            NQ, NK,
            scale,
            D=D,
            Q_TILE=Q_TILE,
            K_TILE=K_TILE,
            is_causal=is_causal,
            num_warps=4,
        )

        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None