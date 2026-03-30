---
name: Centered-B breakthrough for ViT dead neuron revival
description: Mean-centering B across neuron dim enables B-channel gradients for dead neurons, breaking bilinear symmetry. Key discovery for making weight decomposition work with normalized architectures (RMSNorm/LayerNorm).
type: project
---

Mean-centering B in the decomposition `W_eff = W + (B - mean(B, dim=0)) @ C` creates cross-row gradient coupling that the standard bilinear `W + BC` lacks.

**Why:** In standard `W + BC`, the B-gradient for dead neuron j is `G[j,:] Cᵀ = 0` (B-channel dead). With centering, the gradient becomes `-(1/m) Σ_{k alive} G[k,:] Cᵀ ≠ 0`. Dead neurons receive the negative mean of alive neuron gradients through B.

**How to apply:** Use `CenteredDecomposedLinear` (in vit.py) for transformer FFN layers. Standard `DecomposedLinear` works fine for MLPs (no normalization). The centering is needed specifically to overcome the input-normalization bottleneck (RMSNorm/LayerNorm) that makes C-channel-only coupling too weak.

**Results so far:** ViT fc1 block 3 went from 64→53 dead in 20 epochs (11 revived). Later blocks revive faster (closer to loss = stronger gradients). Earlier blocks (B1, B2) are slower. LayerNorm completely blocked revival; RMSNorm + centered-B enables it.

**Open questions:** Does centering also accelerate MLP convergence? What other nonlinearities from paper Section 4.3 could be stronger?
