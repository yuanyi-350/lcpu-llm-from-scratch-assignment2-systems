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
        loop_end = min(num_k_tiles, causal_k_limit)
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

    q = tl.load(Q_block_ptr) 

    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    if is_causal:
        q_idx = q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for kj in range(0, loop_end):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.dot(q, tl.trans(k))
        s = qk * scale

        # 2. Causal Masking (Selective)
        if is_causal:
            if kj == loop_end - 1:
                k_idx = kj * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                mask = q_idx[:, None] < k_idx[None, :]
                s = tl.where(mask, -float("inf"), s)
        
        rowmax = tl.max(s, axis=1)
        m_new = tl.maximum(m, rowmax)
        
        p = tl.exp(s - m_new[:, None])
        l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)

        o = o * tl.exp(m - m_new)[:, None]
        
        # 计算当前的加权和: P @ V
        # 关键点：将 P (fp32) 转回 V 的类型 (bf16) 再做 dot，可以利用 Tensor Cores 加速
        p_cast = p.to(v.type.element_ty)
        o = tl.dot(p_cast, v, acc=o)

        m, l = m_new, l_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o = o / l[:, None]
    L_out = m + tl.log(l)

    # 存储结果 (转回原来的 dtype, 例如 bf16)
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_out)



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
    scale,                           # fp32 scalar (maybe python float)
    D: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # grid = (Tq, B)
    q_tile = tl.program_id(0)
    b = tl.program_id(1)

    q_idx = q_tile * Q_TILE + tl.arange(0, Q_TILE)          # (Q_TILE,)
    d = tl.arange(0, D)                                     # (D,)
    mask_q = q_idx < NQ

    # Load Q tile and dO tile (keep bf16 to use tensor core, cast later as needed)
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

    # L and Drow are fp32 buffers (or will be cast to fp32)
    L = tl.load(L_ptr + b * stride_lb + q_idx * stride_lq, mask=mask_q, other=-float("inf")).to(tl.float32)
    Drow = tl.load(Drow_ptr + b * stride_db + q_idx * stride_dqrow, mask=mask_q, other=0.0).to(tl.float32)

    # dQ accumulator in fp32
    dQ_acc = tl.zeros((Q_TILE, D), tl.float32)

    # Bring scale into triton world (scale might be python float)
    # scale_t = tl.full((), scale, tl.float32)

    # loop over k tiles
    num_k_tiles = tl.cdiv(NK, K_TILE)
    for kt in range(0, num_k_tiles):
        k_idx = kt * K_TILE + tl.arange(0, K_TILE)          # (K_TILE,)
        mask_k = k_idx < NK

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

        # S = Q K^T * scale  (compute logits in fp32 for softmax stability)
        # dot requires same dtype operands: Q and K are both bf16 here.
        S = tl.dot(Q, tl.trans(K)) * scale

        if is_causal:
            causal = q_idx[:, None] < k_idx[None, :]         # future => mask out
            S = S + tl.where(causal, -1e6, 0.0).to(tl.float32)

        # P = exp(S - L)
        P = tl.exp(S - L[:, None])                           # fp32 (Q_TILE, K_TILE)

        # dP = dO V^T  (want fp32)
        dP = tl.dot(dO, tl.trans(V)).to(tl.float32)          # fp32 (Q_TILE, K_TILE)

        # dS = P * (dP - Drow)
        dS = P * (dP - Drow[:, None])                        # fp32 (Q_TILE, K_TILE)

        # dQ += dS K * scale
        dS_cast = dS.to(K.dtype)
        dQ_acc += tl.dot(dS_cast, K).to(tl.float32) * scale


    # store dQ (unique writer: no atomic)
    tl.store(
        dQ_ptr + b * stride_dqb + q_idx[:, None] * stride_dqq + d[None, :] * stride_dqd,
        dQ_acc,
        mask=mask_q[:, None],
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

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + b * stride_qb,
        shape=(NQ, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE, D),
        order=(1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + b * stride_dob,
        shape=(NQ, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE, D),
        order=(1, 0),
    )

    dK_acc = tl.zeros((K_TILE, D), tl.float32)
    dV_acc = tl.zeros((K_TILE, D), tl.float32)

    # loop over q tiles
    num_q_tiles = tl.cdiv(NQ, Q_TILE)
    for qt in range(0, num_q_tiles):
        q_idx = qt * Q_TILE + tl.arange(0, Q_TILE)
        mask_q = q_idx < NQ

        Q = tl.load(Q_block_ptr)
        dO = tl.load(dO_block_ptr)

        L = tl.load(L_ptr + b * stride_lb + q_idx * stride_lq, mask=mask_q, other=-float("inf")).to(tl.float32)
        Drow = tl.load(Drow_ptr + b * stride_db + q_idx * stride_dqrow, mask=mask_q, other=0.0).to(tl.float32)

        S = tl.dot(Q, tl.trans(K), out_dtype=tl.float32) * scale

        if is_causal:
            causal = q_idx[:, None] < k_idx[None, :]
            S = S + tl.where(causal, -1e6, 0.0).to(tl.float32)

        P = tl.exp(S - L[:, None])                           # fp32 (Q_TILE, K_TILE)

        # dV += P^T dO
        # Cast P to V dtype for faster matmul, but accumulate fp32
        P_cast = P.to(V.dtype)
        dV_acc += tl.dot(tl.trans(P_cast), dO).to(tl.float32)

        # dP = dO V^T
        dP = tl.dot(dO, tl.trans(V), out_dtype=tl.float32)   # fp32
        dS = P * (dP - Drow[:, None])                        # fp32

        # dK += dS^T Q * scale
        dS_cast = dS.to(Q.dtype)
        dK_acc += tl.dot(tl.trans(dS_cast), Q) * scale
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE, 0))

    # store dK/dV (unique writer: no atomic)
    tl.store(
        dK_ptr + b * stride_dkb + k_idx[:, None] * stride_dkk + d[None, :] * stride_dkd,
        dK_acc,
        mask=mask_k[:, None],
    )
    tl.store(
        dV_ptr + b * stride_dvb + k_idx[:, None] * stride_dvk + d[None, :] * stride_dvd,
        dV_acc,
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
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        # saved: L, Q, K, V, O
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = getattr(ctx, "is_causal", False)

        B, NQ, D = Q.shape
        _, NK, _ = K.shape
        scale = ctx.scale

        # tiles
        Q_TILE = 64
        K_TILE = 64
        assert NQ % Q_TILE == 0 and NK % K_TILE == 0 and D >= 16

        # Dvec = rowsum(dO * O), fp32
        Dvec = torch.empty((B, NQ), device=Q.device, dtype=torch.float32)

        # grads in fp32 (then cast back to input dtype at return)
        dQ = torch.empty((B, NQ, D), device=Q.device, dtype=torch.float32)
        dK = torch.empty((B, NK, D), device=Q.device, dtype=torch.float32)
        dV = torch.empty((B, NK, D), device=Q.device, dtype=torch.float32)

        # (optional but safe) initialize if your kernels fully overwrite outputs
        # dQ.zero_(); dK.zero_(); dV.zero_()
        # 这里拆分后是 unique-writer + tl.store，理论上不需要 zero_

        # 1) compute D = rowsum(dO ∘ O)
        grid_D = (NQ // Q_TILE, B)
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

        # 2a) dQ kernel: grid = (Tq, B)
        grid_dq = (NQ // Q_TILE, B)
        flash_bwd_dq_kernel[grid_dq](
            Q, K, V,
            dO,
            L, Dvec,
            dQ,
            # Q strides
            Q.stride(0), Q.stride(1), Q.stride(2),
            # K strides
            K.stride(0), K.stride(1), K.stride(2),
            # V strides
            V.stride(0), V.stride(1), V.stride(2),
            # dO strides
            dO.stride(0), dO.stride(1), dO.stride(2),
            # L strides
            L.stride(0), L.stride(1),
            # Dvec strides
            Dvec.stride(0), Dvec.stride(1),
            # dQ strides
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            # runtime args
            NQ=NQ, NK=NK,
            scale=scale,
            # constexpr args
            D=D,
            Q_TILE=Q_TILE,
            K_TILE=K_TILE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=3,
        )

        # 2b) dK/dV kernel: grid = (Tk, B)
        grid_dkv = (NK // K_TILE, B)
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
            Q_TILE=Q_TILE,
            K_TILE=K_TILE,
            is_causal=is_causal,
            num_warps=4,
            num_stages=3,
        )

        # cast back to input dtype to match autograd expectations
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None
