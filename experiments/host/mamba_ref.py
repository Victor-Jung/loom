#!/usr/bin/env python
"""Reference implementations of the Loom mamba_chunk_scan kernel.

Two independent implementations (vectorised + naive nested loops) that must
agree, so the reference itself is trustworthy before it is ever compared to
device output.

NOTE ON SEMANTICS: kernels/mamba_chunk_scan.py has the causal mask commented
out ("Yet not support sparse matmul") and its k-loop runs to
(tile_m.id + 1) * block_m.  The kernel therefore computes a BLOCK-causal result
that includes k > m terms inside m's own block, and the answer depends on
block_m.  `causal="element"` is the true Mamba-2 chunk scan; `causal="block"` is the
original kernel; `causal="none"` is what the kernel computes once the dynamic
trip count is replaced by a static chunk_size.
"""
import torch

def mamba_ref(cb, x, dt, dA, C, prev_states, D, block_m, causal="block"):
    """Vectorised. Shapes per the kernel docstring; returns [B, H, S, Dh]."""
    Bb, NC, G, CS, _ = cb.shape
    _, S, H, Dh = x.shape
    dstate = C.shape[-1]
    xt = x.transpose(1, 2).float()                   # [B,H,S,Dh]
    Ct = C.transpose(1, 2).float()                   # [B,G,S,dstate]
    ps = prev_states.float()                         # [B,NC,H,Dh,dstate]
    out = torch.zeros(Bb, H, S, Dh, dtype=torch.float32)
    hpg = H // G
    m_idx = torch.arange(CS)
    for c in range(NC):
        for hh in range(H):
            g = hh // hpg
            dA_c = dA[:, hh, c, :].float()           # [B,CS]
            dt_c = dt[:, hh, c, :].float()           # [B,CS]
            Cl = Ct[:, g, c*CS:(c+1)*CS, :]          # [B,CS,dstate]
            # state term: exp(dA[m]) * (C[m,:] @ prev_states[..,n,:])
            st = torch.einsum('bmd,bnd->bmn', Cl, ps[:, c, hh])          # [B,CS,Dh]
            acc = st * torch.exp(dA_c).unsqueeze(-1)
            # intra-chunk term
            cbl = cb[:, c, g].float()                                     # [B,CS,CS]
            decay = torch.exp(dA_c.unsqueeze(2) - dA_c.unsqueeze(1))      # [B,CS(m),CS(k)]
            W = cbl * decay * dt_c.unsqueeze(1)                           # [B,m,k]
            if causal == "element":
                mask = m_idx.unsqueeze(1) >= m_idx.unsqueeze(0)
                W = W * mask.to(W.dtype)
            elif causal == "block":
                kend = ((m_idx // block_m) + 1) * block_m
                mask = m_idx.unsqueeze(0) < kend.unsqueeze(1)             # [m,k]
                W = W * mask.to(W.dtype)
            elif causal == "none":
                # What the kernel computes after the dynamic trip count
                # (tile_m.id + 1) * block_m was replaced by a static chunk_size:
                # every k in the chunk contributes, so there is no mask at all.
                pass
            else:
                raise ValueError(f"unknown causal mode {causal!r}")
            xc = xt[:, hh, c*CS:(c+1)*CS, :]                              # [B,CS,Dh]
            acc = acc + torch.einsum('bmk,bkn->bmn', W, xc)
            acc = acc + xc * D[hh].float()
            out[:, hh, c*CS:(c+1)*CS, :] = acc
    return out

def mamba_ref_naive(cb, x, dt, dA, C, prev_states, D, block_m, causal="block"):
    """Explicit loops; slow, used only to cross-check mamba_ref."""
    Bb, NC, G, CS, _ = cb.shape
    _, S, H, Dh = x.shape
    xt = x.transpose(1, 2).float(); Ct = C.transpose(1, 2).float(); ps = prev_states.float()
    out = torch.zeros(Bb, H, S, Dh, dtype=torch.float32)
    hpg = H // G
    for b in range(Bb):
        for c in range(NC):
            for hh in range(H):
                g = hh // hpg
                for m in range(CS):
                    kend = m + 1 if causal == "element" else ((m // block_m) + 1) * block_m
                    for n in range(Dh):
                        v = float(torch.dot(Ct[b, g, c*CS+m, :], ps[b, c, hh, n, :]))
                        v *= float(torch.exp(dA[b, hh, c, m].float()))
                        s = 0.0
                        for k in range(min(kend, CS)):
                            s += (float(cb[b, c, g, m, k]) *
                                  float(torch.exp(dA[b, hh, c, m].float() - dA[b, hh, c, k].float())) *
                                  float(dt[b, hh, c, k]) * float(xt[b, hh, c*CS+k, n]))
                        v += s + float(xt[b, hh, c*CS+m, n]) * float(D[hh])
                        out[b, hh, c*CS+m, n] = v
    return out
