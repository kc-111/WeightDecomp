"""
Two-layer MLP fitting linear data with Hurwitz-inspired overparameterization.

Data: Y = X @ W_star.T  (linear)
Model: Y_hat = W2_eff @ ReLU(X @ W1_eff.T)  (nonlinear)

Each W_eff = W + A @ sum_k(B_k B_k^T + C_k - C_k^T) @ D
  B_k is N x r  → B_k B_k^T is symmetric PSD, rank r
  C_k - C_k^T is skew-symmetric (C parameterized as strict lower triangular)
  K components are summed inside the sandwich
  A is N x N (randomly initialized)
  D is N x N_in (zero-initialized, so correction starts at zero)

Compared against baseline and plain W+BC from linear_test.py.

Usage: python hurwitz_decomp_test.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ============================================================
# Problem setup (same as linear_test.py)
# ============================================================

d_in = 128
d_hid = 128
d_out = 128
num_samples = 1024
num_steps = 2000

# Ill-conditioned data (condition number ~1e5)
singular_values = torch.logspace(0, -5, d_in)
U_data = torch.linalg.qr(torch.randn(d_in, d_in))[0]
sqrt_cov = U_data @ torch.diag(singular_values)
X = torch.randn(num_samples, d_in) @ sqrt_cov.T
W_star = torch.randn(d_out, d_in) * 0.1
Y = X @ W_star.T


def compute_loss(Y_hat, Y):
    return 0.5 * ((Y_hat - Y) ** 2).mean()


# ============================================================
# Helpers
# ============================================================

def flat_to_strict_tril(flat, N):
    """Place flat vector [N*(N-1)/2] into strict lower-triangular NxN matrix."""
    idx = torch.tril_indices(N, N, offset=-1, device=flat.device)
    flat_pos = idx[0] * N + idx[1]
    result = torch.zeros(N * N, dtype=flat.dtype, device=flat.device)
    result = result.scatter(0, flat_pos, flat)
    return result.reshape(N, N)


# ============================================================
# Training functions
# ============================================================

def train_baseline(W1_init, W2_init, lr, num_steps, alpha=0.1):
    W1 = torch.nn.Parameter(W1_init.clone() * alpha)
    W2 = torch.nn.Parameter(W2_init.clone() * alpha)
    optimizer = torch.optim.Adam([W1, W2], lr=lr)
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        Y_hat = torch.relu(X @ W1.T) @ W2.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [W1, W2]):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()
    return losses


def train_BC(W1_init, W2_init, B1_init, B2_init, lr, num_steps, alpha=0.1):
    """W_eff = alpha*W + B @ C per layer, all trainable."""
    r1, r2 = B1_init.shape[1], B2_init.shape[1]
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    B1 = torch.nn.Parameter(B1_init.clone())
    C1 = torch.nn.Parameter(torch.zeros(r1, d_in))
    B2 = torch.nn.Parameter(B2_init.clone())
    C2 = torch.nn.Parameter(torch.zeros(r2, d_hid))
    params = [W1, W2, B1, C1, B2, C2]
    optimizer = torch.optim.Adam(params, lr=lr)
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        W1_eff = alpha * W1 + B1 @ C1
        W2_eff = alpha * W2 + B2 @ C2
        Y_hat = torch.relu(X @ W1_eff.T) @ W2_eff.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()
    return losses


def train_hurwitz_decomp(W1_init, W2_init, r, K, lr, num_steps, alpha=0.1):
    """W_eff = alpha*W + A @ sum_k(B_k B_k^T + Skew_k) @ D per layer."""
    N1, N1_in = W1_init.shape
    N2, N2_in = W2_init.shape
    comp_scale = 0.1 / np.sqrt(K)

    # Layer 1
    W1 = torch.nn.Parameter(W1_init.clone())
    B1 = torch.nn.Parameter(torch.randn(K, N1, r) * comp_scale)
    M1_flat = torch.nn.Parameter(torch.randn(K, N1 * (N1 - 1) // 2) * comp_scale)
    A1 = torch.nn.Parameter(torch.randn(N1, N1) * 0.1)
    D1 = torch.nn.Parameter(torch.zeros(N1, N1_in))

    # Layer 2
    W2 = torch.nn.Parameter(W2_init.clone())
    B2 = torch.nn.Parameter(torch.randn(K, N2, r) * comp_scale)
    M2_flat = torch.nn.Parameter(torch.randn(K, N2 * (N2 - 1) // 2) * comp_scale)
    A2 = torch.nn.Parameter(torch.randn(N2, N2) * 0.1)
    D2 = torch.nn.Parameter(torch.zeros(N2, N2_in))

    params = [W1, B1, M1_flat, A1, D1, W2, B2, M2_flat, A2, D2]
    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        inner1 = torch.zeros(N1, N1)
        for k in range(K):
            Skew = flat_to_strict_tril(M1_flat[k], N1)
            inner1 = inner1 + B1[k] @ B1[k].T + Skew - Skew.T
        W1_eff = alpha * W1 + A1 @ inner1 @ D1

        inner2 = torch.zeros(N2, N2)
        for k in range(K):
            Skew = flat_to_strict_tril(M2_flat[k], N2)
            inner2 = inner2 + B2[k] @ B2[k].T + Skew - Skew.T
        W2_eff = alpha * W2 + A2 @ inner2 @ D2

        Y_hat = torch.relu(X @ W1_eff.T) @ W2_eff.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()

    return losses


# ============================================================
# Run experiments
# ============================================================

W1_init = torch.randn(d_hid, d_in) * 0.5
W2_init = torch.randn(d_out, d_hid) * 0.5

r_values = [4, 16, 64]
K_values = [4]
lr_candidates = [10.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]

# Baseline
print("Tuning baseline lr...")
best_base_lr, best_base_loss = None, float('inf')
for lr in lr_candidates:
    losses = train_baseline(W1_init, W2_init, lr, num_steps)
    if losses[-1] < best_base_loss:
        best_base_loss = losses[-1]
        best_base_lr = lr
print(f"  Best lr={best_base_lr}, final loss={best_base_loss:.6f}")
losses_base = train_baseline(W1_init, W2_init, best_base_lr, num_steps)

# BC (scaled down to match Hurwitz init: 0.1 scale)
results_bc = {}
for r in r_values:
    B1_init = torch.randn(d_hid, r) * 0.1
    B2_init = torch.randn(d_out, r) * 0.1
    print(f"Tuning lr for BC r={r}...")
    best_lr, best_loss = None, float('inf')
    for lr in lr_candidates:
        losses = train_BC(W1_init, W2_init, B1_init, B2_init, lr, num_steps)
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_lr = lr
    print(f"  Best lr={best_lr}, final loss={best_loss:.6f}")
    results_bc[r] = (train_BC(W1_init, W2_init, B1_init, B2_init, best_lr, num_steps), best_lr)

# Hurwitz decomp: sweep (r, K)
results_hd = {}
for r in r_values:
    for K in K_values:
        print(f"Tuning lr for Hurwitz decomp r={r}, K={K}...")
        best_lr, best_loss = None, float('inf')
        for lr in lr_candidates:
            losses = train_hurwitz_decomp(W1_init, W2_init, r, K, lr, num_steps)
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_lr = lr
        print(f"  Best lr={best_lr}, final loss={best_loss:.6f}")
        results_hd[(r, K)] = (train_hurwitz_decomp(W1_init, W2_init, r, K, best_lr, num_steps), best_lr)


# ============================================================
# Plot results
# ============================================================

hd_configs = [(r, K) for r in r_values for K in K_values]
n_configs = len(hd_configs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
r_labels = [str(r) for r in r_values]
x_pos = np.arange(len(r_values))

# --- Plot 1: Loss curves (all HD configs vs baseline) ---
ax = axes[0]
ax.semilogy(losses_base, 'k-', linewidth=2, label=f'Baseline (lr={best_base_lr})')
linestyles = {1: '-', 4: '--'}
for r in r_values:
    for K in K_values:
        losses, lr = results_hd[(r, K)]
        ax.semilogy(losses, linestyle=linestyles[K], alpha=0.8,
                    label=f'r={r} K={K} (lr={lr})')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('W + A @ sum_k(B_k B_k^T + Skew_k) @ D')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

# --- Plot 2: Loss curves (HD K=4 vs BC) ---
ax = axes[1]
ax.semilogy(losses_base, 'k-', linewidth=2, label='Baseline')
for r in r_values:
    ax.semilogy(results_bc[r][0], '-', alpha=0.7, label=f'BC r={r}')
    ax.semilogy(results_hd[(r, 4)][0], '--', alpha=0.7, label=f'HD r={r} K=4')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('BC vs Hurwitz Decomp (K=4)')
ax.legend(fontsize=5, ncol=2)
ax.grid(True, alpha=0.3)

# --- Plot 3: Final loss bar chart ---
ax = axes[2]
n_bars = 1 + len(K_values)  # BC + one per K
width = 0.8 / n_bars
for r_idx, r in enumerate(r_values):
    # BC bar
    ax.bar(x_pos[r_idx] - width, results_bc[r][0][-1], width,
           color='tab:blue', label='W+BC' if r_idx == 0 else '')
    # HD bars per K
    K_colors = {1: 'tab:orange', 4: 'tab:red'}
    for k_idx, K in enumerate(K_values):
        ax.bar(x_pos[r_idx] + k_idx * width, results_hd[(r, K)][0][-1], width,
               color=K_colors[K],
               label=f'HD K={K}' if r_idx == 0 else '')
ax.axhline(losses_base[-1], color='k', linestyle='--', label='Baseline')
ax.set_xticks(x_pos)
ax.set_xticklabels(r_labels)
ax.set_xlabel('Rank r')
ax.set_ylabel('Final Loss')
ax.set_yscale('log')
ax.set_title('Final Loss vs Rank')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hurwitz_decomp_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary table
# ============================================================

def compute_speedup(losses, target):
    for i, l in enumerate(losses):
        if l <= target:
            return num_steps / i if i > 0 else float('inf')
    return num_steps / num_steps

print("\n" + "=" * 85)
print(f"{'Condition':<35s} {'Best LR':>8s} {'Final Loss':>12s} {'Speedup':>10s}")
print("=" * 85)
print(f"{'Baseline':<35s} {best_base_lr:>8.4f} {losses_base[-1]:>12.6f} {'1.00x':>10s}")

for r in r_values:
    losses, lr = results_bc[r]
    speedup = compute_speedup(losses, losses_base[-1])
    print(f"{'BC r=' + str(r):<35s} {lr:>8.4f} {losses[-1]:>12.6f} {speedup:>9.1f}x")

for r in r_values:
    for K in K_values:
        losses, lr = results_hd[(r, K)]
        speedup = compute_speedup(losses, losses_base[-1])
        label = f"HD r={r} K={K}"
        print(f"{label:<35s} {lr:>8.4f} {losses[-1]:>12.6f} {speedup:>9.1f}x")

print("=" * 85)
