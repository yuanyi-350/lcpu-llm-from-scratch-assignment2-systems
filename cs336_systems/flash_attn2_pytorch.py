import math
import torch
from einops import einsum



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
    def backward(ctx, dO):
        """
        Implements Algorithm 2 (tiled FlashAttention-2 backward) in pure PyTorch.

        Inputs:
          dO: gradient wrt output O, shape (B, Nq, D)

        Saved (from forward):
          Q: (B, Nq, D)
          K: (B, Nk, D)
          V: (B, Nk, D)
          O: (B, Nq, D)
          L: (B, Nq)  # row-wise logsumexp of S
        """
        # Unpack saved tensors
        L, Q, K, V, O = ctx.saved_tensors
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        assert D == Dk and dO.shape == O.shape

        # Tile sizes (>=16). You can tune later.
        Bq = 16
        Bk = 16

        scale = 1.0 / math.sqrt(D)

        # Algorithm 2: D_vec = rowsum(dO ⊙ O)  (per query row)
        # Shape: (B, Nq)
        D_vec = torch.sum(dO.to(torch.float32) * O.to(torch.float32), dim=-1)

        # Allocate grads
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K, dtype=torch.float32)
        dV = torch.zeros_like(V, dtype=torch.float32)

        # Split K,V into key tiles; outer loop over j (as in Algorithm 2)
        for b in range(B):
            for kj in range(0, Nk, Bk):
                k = K[b, kj:kj+Bk, :].to(torch.float32)  # (Bk, D)
                v = V[b, kj:kj+Bk, :].to(torch.float32)  # (Bk, D)

                dK_j = torch.zeros_like(k)  # (Bk, D)
                dV_j = torch.zeros_like(v)  # (Bk, D)

                # Inner loop over query tiles i
                for qi in range(0, Nq, Bq):
                    q = Q[b, qi:qi+Bq, :].to(torch.float32)      # (Bq, D)
                    do = dO[b, qi:qi+Bq, :].to(torch.float32)    # (Bq, D)
                    l = L[b, qi:qi+Bq].to(torch.float32)         # (Bq,)
                    d_row = D_vec[b, qi:qi+Bq].to(torch.float32) # (Bq,)

                    S = einsum(q, k, "q d, k d -> q k") * scale  # (Bq, Bk)
                    P = torch.exp(S - l[:, None])                # (Bq, Bk)

                    dV_j += einsum(P, do, "q k, q d -> k d")     # (Bk, D)
                    dP = einsum(do, v, "q d, k d -> q k")        # (Bq, Bk)
                    dS = P * (dP - d_row[:, None])               # (Bq, Bk)

                    dQ[b, qi:qi+Bq, :] += einsum(dS, k, "q k, k d -> q d") * scale # (Bq, D)
                    dK_j += einsum(dS, q, "q k, q d -> k d") * scale               # (Bk, D)

                dK[b, kj:kj+Bk, :] += dK_j
                dV[b, kj:kj+Bk, :] += dV_j

        # cast grads back to input dtype
        dQ = dQ.to(Q.dtype)
        dK = dK.to(K.dtype)
        dV = dV.to(V.dtype)

        # backward signature must match forward inputs: (Q, K, V, is_causal)
        return dQ, dK, dV, None
