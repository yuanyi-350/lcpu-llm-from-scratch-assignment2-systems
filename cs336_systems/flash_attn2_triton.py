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
    q_tile = tl.program_id(0)
    b = tl.program_id(1)

    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    if is_causal:
        causal_k_limit = tl.cdiv((q_tile + 1) * Q_TILE_SIZE, K_TILE_SIZE)
        loop_end = tl.minimum(num_k_tiles, causal_k_limit)
    else:
        loop_end = num_k_tiles

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + b * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + b * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + b * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + b * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + b * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    q = tl.load(Q_block_ptr, boundary_check=(0,))

    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    if is_causal:
        q_idx = q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for kj in range(0, loop_end):
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        s = qk * scale

        if is_causal:
            k_idx = kj * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            s = tl.where(mask, -float("inf"), s)
        
        rowmax = tl.max(s, axis=1)
        m_new = tl.maximum(m, rowmax)
        
        p = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)

        o = o * tl.exp(m - m_new)[:, None]
        
        # 计算当前的加权和: P @ V
        # [KeyPoint] 将 P (fp32) 转回 V 的类型 (bf16) 再做 dot，可以利用 Tensor Cores 加速
        p_cast = p.to(v.type.element_ty)
        o = tl.dot(p_cast, v, acc=o)

        m, l = m_new, l_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o = o / l[:, None]
    L_out = m + tl.log(l)

    # 存储结果 (转回原来的 dtype, 例如 bf16)
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L_out, boundary_check=(0,))



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
    )
    O = tl.load(
        O_ptr + b * stride_ob + q_idx[:, None] * stride_oq + d[None, :] * stride_od,
        mask=mask_q[:, None],
        other=0.0,
    )

    # rowsum over D
    D_row = tl.sum(dO * O, axis=1)  # (Q_TILE,)

    tl.store(
        D_ptr + b * stride_db + q_idx * stride_dq,
        D_row,
        mask=mask_q
    )



@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr,
    L_ptr, Drow_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dqrow,
    stride_dqb, stride_dqq, stride_dqd,
    NQ: tl.constexpr, NK: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # grid = (Tq, B)
    q_tile = tl.program_id(0)
    b = tl.program_id(1)

    q_idx = q_tile * Q_TILE + tl.arange(0, Q_TILE)
    mask_q = q_idx < NQ
    d = tl.arange(0, D)

    Q = tl.load(
        Q_ptr + b * stride_qb + q_idx[:, None] * stride_qq + d[None, :] * stride_qd,
        mask=mask_q[:, None],
        other=0.0,
    )
    dO = tl.load(
        dO_ptr + b * stride_dob + q_idx[:, None] * stride_doq + d[None, :] * stride_dod,
        mask=mask_q[:, None],
        other=0.0,
    )

    L = tl.load(L_ptr + b * stride_lb + q_idx * stride_lq, mask=mask_q, other=-float("inf")).to(tl.float32)
    Drow = tl.load(Drow_ptr + b * stride_db + q_idx * stride_dqrow, mask=mask_q, other=0.0).to(tl.float32)

    dQ_acc = tl.zeros((Q_TILE, D), tl.float32)

    # [KeyPoint] Causal 模式下，动态计算 K 循环的终点
    loop_end = tl.cdiv(NK, K_TILE)
    if is_causal:
        loop_end = tl.minimum(loop_end, tl.cdiv((q_tile + 1) * Q_TILE, K_TILE))

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + b * stride_kb,
        shape=(NK, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + b * stride_vb,
        shape=(NK, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE, D),
        order=(1, 0)
    )

    for kt in range(0, loop_end):
        K = tl.load(K_block_ptr, boundary_check=(0,))
        V = tl.load(V_block_ptr, boundary_check=(0,))

        S = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * scale
        P = tl.exp(S - L[:, None])

        if is_causal:
            k_idx = kt * K_TILE + tl.arange(0, K_TILE)
            causal_mask = q_idx[:, None] >= k_idx[None, :] # True 代表可见
            P = tl.where(causal_mask, P, 0.0)

        dP = tl.dot(dO, tl.trans(V), out_dtype=tl.float32)
        dS = P * (dP - Drow[:, None])

        dQ_acc += tl.dot(dS.to(K.dtype), K, out_dtype=tl.float32) * scale

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE, 0))

    tl.store(
        dQ_ptr + b * stride_dqb + q_idx[:, None] * stride_dqq + d[None, :] * stride_dqd,
        dQ_acc.to(Q.dtype),
        mask=mask_q[:, None],
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"Q_TILE": Q_TILE, "K_TILE": K_TILE},
            # num_warps=4,  # ✅ 正确写法：在这里显式指定
            # num_stages=3  # ✅ 正确写法：在这里显式指定
        )
        for Q_TILE in [32, 64]
        for K_TILE in [64, 128]
    ],
    key=["NQ", "NK", "D", "is_causal"],
)
@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr,
    L_ptr, Drow_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dqrow,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    NQ: tl.constexpr, NK: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # grid = (Tk, B)
    k_tile = tl.program_id(0)
    b = tl.program_id(1)

    k_idx = k_tile * K_TILE + tl.arange(0, K_TILE)          # (K_TILE,)
    d = tl.arange(0, D)
    mask_k = k_idx < NK

    # Load K/V tile (bf16)
    K = tl.load(
        K_ptr + b * stride_kb + k_idx[:, None] * stride_kk + d[None, :] * stride_kd,
        mask=mask_k[:, None],
        other=0.0,
    )
    V = tl.load(
        V_ptr + b * stride_vb + k_idx[:, None] * stride_vk + d[None, :] * stride_vd,
        mask=mask_k[:, None],
        other=0.0,
    )

    dK_acc = tl.zeros((K_TILE, D), tl.float32)
    dV_acc = tl.zeros((K_TILE, D), tl.float32)

    # loop over q tiles
    num_q_tiles = tl.cdiv(NQ, Q_TILE)

    start_q_tile = (k_tile * K_TILE) // Q_TILE if is_causal else 0

    start_q_tile = (k_tile * K_TILE) // Q_TILE if is_causal else 0
    start_q_offset = start_q_tile * Q_TILE
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + b * stride_qb,
        shape=(NQ, D),
        strides=(stride_qq, stride_qd),
        offsets=(start_q_offset, 0),  # <--- 必须从这里开始！
        block_shape=(Q_TILE, D),
        order=(1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + b * stride_dob,
        shape=(NQ, D),
        strides=(stride_doq, stride_dod),
        offsets=(start_q_offset, 0),  # <--- 必须从这里开始！
        block_shape=(Q_TILE, D),
        order=(1, 0),
    )

    # [keyPoint]
    for qt in range(start_q_tile, num_q_tiles):
        q_idx = qt * Q_TILE + tl.arange(0, Q_TILE)
        mask_q = q_idx < NQ

        Q = tl.load(Q_block_ptr, boundary_check=(0,))
        dO = tl.load(dO_block_ptr, boundary_check=(0,))

        L = tl.load(L_ptr + b * stride_lb + q_idx * stride_lq, mask=mask_q, other=-float("inf"))
        Drow = tl.load(Drow_ptr + b * stride_db + q_idx * stride_dqrow, mask=mask_q, other=0.0)

        S = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * scale

        P = tl.exp(S - L[:, None])
        if is_causal:
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            P = tl.where(causal_mask, P, 0.0)

        # dV += P^T dO
        # [KeyPoint]: Cast P to V dtype for faster matmul, but accumulate fp32
        P_cast = P.to(V.dtype)
        dV_acc += tl.dot(tl.trans(P_cast), dO, out_dtype=tl.float32)

        # dP = dO V^T
        dP = tl.dot(dO, tl.trans(V), out_dtype=tl.float32)   # fp32
        dS = P * (dP - Drow[:, None])                        # fp32

        # dK += dS^T Q * scale
        dS_cast = dS.to(Q.dtype)
        dK_acc += tl.dot(tl.trans(dS_cast), Q, out_dtype=tl.float32) * scale
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE, 0))

    # store dK/dV (unique writer: no atomic)
    tl.store(
        dK_ptr + b * stride_dkb + k_idx[:, None] * stride_dkk + d[None, :] * stride_dkd,
        dK_acc.to(K.dtype),
        mask=mask_k[:, None],
    )
    tl.store(
        dV_ptr + b * stride_dvb + k_idx[:, None] * stride_dvk + d[None, :] * stride_dvd,
        dV_acc.to(V.dtype),
        mask=mask_k[:, None],
    )



class FlashAttn2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        assert D == Dk and V.shape[:2] == (B, Nk)

        # [KeyPoint]
        Q_TILE = 64
        K_TILE = 64
        scale = 1.0 / math.sqrt(D)

        O = torch.empty((B, Nq, D), device=Q.device, dtype=Q.dtype)
        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(Nq, Q_TILE), B)

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
            num_warps=4,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = getattr(ctx, "is_causal", False)

        B, NQ, D = Q.shape
        _, NK, _ = K.shape
        scale = 1.0 / math.sqrt(D)

        # [KeyPoint]
        Q_TILE = 64
        K_TILE = 64
        assert NQ % Q_TILE == 0 and NK % K_TILE == 0 and D >= 16

        Dvec = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        # grads in fp32 (then cast back to input dtype at return)
        dQ = torch.empty((B, NQ, D), device=Q.device, dtype=torch.float32)
        dK = torch.empty((B, NK, D), device=Q.device, dtype=torch.float32)
        dV = torch.empty((B, NK, D), device=Q.device, dtype=torch.float32)

        grid_D = (triton.cdiv(NQ, Q_TILE), B)
        compute_D_kernel[grid_D](
            dO, O, Dvec,
            dO.stride(0), dO.stride(1), dO.stride(2),
            O.stride(0),  O.stride(1),  O.stride(2),
            Dvec.stride(0), Dvec.stride(1),
            NQ,
            D=D,
            Q_TILE=Q_TILE,
            num_warps=4,
        )

        grid_dq = (triton.cdiv(NQ, Q_TILE), B)
        flash_bwd_dq_kernel[grid_dq](
            Q, K, V,
            dO,
            L, Dvec,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),

            Dvec.stride(0), Dvec.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),

            NQ=NQ, NK=NK,
            scale=scale,

            D=D,
            Q_TILE=Q_TILE,
            K_TILE=K_TILE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=3,
        )

        # 2b) dK/dV kernel: grid = (Tk, B)
        grid_dkv = lambda META: (triton.cdiv(NK, META["K_TILE"]), B)
        flash_bwd_dkv_kernel[grid_dkv](
            Q, K, V,
            dO,
            L, Dvec,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            Dvec.stride(0), Dvec.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            NQ=NQ, NK=NK,
            scale=scale,
            D=D,
            # Q_TILE=Q_TILE,
            # K_TILE=K_TILE,
            is_causal=is_causal,
            # num_warps=4,
            # num_stages=3,
        )

        # cast back to input dtype to match autograd expectations
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None
