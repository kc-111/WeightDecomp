"""
Two-layer MLP fitting linear data.
Data: Y = X @ W_star.T  (linear)
Model: Y_hat = W2_eff @ ReLU(X @ W1_eff.T)  (nonlinear)
Each W_eff = W + BC  (and optionally + DE with rank-k DE)
BC is local per layer. Vary rank r, tune lr per r.

Key questions:
1. How does rank r affect convergence?
2. Does overparameterized r (r > n) help?
3. Does adding a second low-rank factor DE help?

Usage: python linear_test.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# ============================================================
# Problem setup
# ============================================================

d_in = 128    # input dim
d_hid = 128   # hidden dim
d_out = 128   # output dim
k = 1         # rank of second factor DE
num_samples = 1024
num_steps = 2000

# Ill-conditioned data (condition number ~1e5)
singular_values = torch.logspace(0, -5, d_in)
U_data = torch.linalg.qr(torch.randn(d_in, d_in))[0]
sqrt_cov = U_data @ torch.diag(singular_values)
X = torch.randn(num_samples, d_in) @ sqrt_cov.T
W_star = torch.randn(d_out, d_in) * 0.1
Y = X @ W_star.T  # linear data


def forward_baseline(X, W1, W2):
    return torch.relu(X @ W1.T) @ W2.T


def compute_loss(Y_hat, Y):
    return 0.5 * ((Y_hat - Y) ** 2).mean()


# ============================================================
# Training functions
# ============================================================

def train_baseline(W1_init, W2_init, lr, num_steps):
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    optimizer = torch.optim.Adam([W1, W2], lr=lr)
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        Y_hat = forward_baseline(X, W1, W2)
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [W1, W2]):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()
    return losses


def train_BC(W1_init, W2_init, B1_init, B2_init, lr, num_steps):
    """W1_eff = W1 + B1@C1, W2_eff = W2 + B2@C2, all trainable."""
    r1 = B1_init.shape[1]
    r2 = B2_init.shape[1]
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    B1 = torch.nn.Parameter(B1_init.clone())
    B2 = torch.nn.Parameter(B2_init.clone())
    C1 = torch.nn.Parameter(torch.zeros(r1, d_in))
    C2 = torch.nn.Parameter(torch.zeros(r2, d_hid))
    optimizer = torch.optim.Adam([W1, W2, B1, B2, C1, C2], lr=lr)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        W1_eff = W1 + B1 @ C1
        W2_eff = W2 + B2 @ C2
        Y_hat = torch.relu(X @ W1_eff.T) @ W2_eff.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [W1, W2, B1, B2, C1, C2]):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()

    return losses


def train_BC_DE(W1_init, W2_init, B1_init, B2_init, D1_init, D2_init, lr, num_steps):
    """W_eff = W + BC + DE per layer, all trainable."""
    r1, r2 = B1_init.shape[1], B2_init.shape[1]
    k1, k2 = D1_init.shape[1], D2_init.shape[1]
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    B1 = torch.nn.Parameter(B1_init.clone())
    B2 = torch.nn.Parameter(B2_init.clone())
    C1 = torch.nn.Parameter(torch.zeros(r1, d_in))
    C2 = torch.nn.Parameter(torch.zeros(r2, d_hid))
    D1 = torch.nn.Parameter(D1_init.clone())
    D2 = torch.nn.Parameter(D2_init.clone())
    E1 = torch.nn.Parameter(torch.zeros(k1, d_in))
    E2 = torch.nn.Parameter(torch.zeros(k2, d_hid))
    params = [W1, W2, B1, B2, C1, C2, D1, D2, E1, E2]
    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        W1_eff = W1 + B1 @ C1 + D1 @ E1
        W2_eff = W2 + B2 @ C2 + D2 @ E2
        Y_hat = torch.relu(X @ W1_eff.T) @ W2_eff.T
        loss = compute_loss(Y_hat, Y)
        losses.append(loss.item())
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params):
            losses.extend([losses[-1]] * (num_steps - step - 1))
            break
        optimizer.step()

    return losses


def get_final_diagnostics(W1_init, W2_init, B1_init, B2_init, lr, num_steps):
    """Run BC training with Adam, return end-of-training diagnostics for layer 1."""
    r = B1_init.shape[1]
    r2 = B2_init.shape[1]
    W1 = torch.nn.Parameter(W1_init.clone())
    W2 = torch.nn.Parameter(W2_init.clone())
    B1 = torch.nn.Parameter(B1_init.clone())
    B2 = torch.nn.Parameter(B2_init.clone())
    C1 = torch.nn.Parameter(torch.zeros(r, d_in))
    C2 = torch.nn.Parameter(torch.zeros(r2, d_hid))
    optimizer = torch.optim.Adam([W1, W2, B1, B2, C1, C2], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        W1_eff = W1 + B1 @ C1
        W2_eff = W2 + B2 @ C2
        Y_hat = torch.relu(X @ W1_eff.T) @ W2_eff.T
        loss = compute_loss(Y_hat, Y)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in [W1, W2, B1, B2, C1, C2]):
            break
        optimizer.step()

    # Final gradient w.r.t. W1_eff via autograd
    W1_eff = (W1 + B1 @ C1).detach().requires_grad_(True)
    W2_det = (W2 + B2 @ C2).detach()
    Y_hat = torch.relu(X @ W1_eff.T) @ W2_det.T
    loss = compute_loss(Y_hat, Y)
    G = torch.autograd.grad(loss, W1_eff)[0]

    with torch.no_grad():
        BC = B1 @ C1

        # Effective rank of B1 B1^T
        eigvals = torch.linalg.eigvalsh(B1 @ B1.T).clamp(min=1e-12)
        p = eigvals / eigvals.sum()
        eff_rank = torch.exp(-torch.sum(p * torch.log(p))).item()

        # BC alignment with gradient SVs
        r_align = min(r, min(d_hid, d_in))
        if torch.isfinite(G).all() and BC.norm() > 1e-12:
            U_G = torch.linalg.svd(G, full_matrices=False)[0]
            BC_norm_sq = BC.norm() ** 2 + 1e-12
            top_align = (U_G[:, :r_align].T @ BC).norm() ** 2 / BC_norm_sq
            bot_align = (U_G[:, -r_align:].T @ BC).norm() ** 2 / BC_norm_sq
        else:
            top_align = torch.tensor(float('nan'))
            bot_align = torch.tensor(float('nan'))

        # Preconditioner condition number
        P_eigs = torch.linalg.eigvalsh(torch.eye(d_hid) + B1 @ B1.T)
        precond_cond = (P_eigs[-1] / P_eigs[0]).item()

        # Column collapse
        if r > 1:
            B_normed = B1 / (B1.norm(dim=0, keepdim=True) + 1e-12)
            cosine_matrix = B_normed.T @ B_normed
            mask = ~torch.eye(r, dtype=bool)
            col_cosine = cosine_matrix[mask].abs().mean().item()
        else:
            col_cosine = 0.0

    return {
        'effective_rank': eff_rank,
        'alignment_top': top_align.item(),
        'alignment_bot': bot_align.item(),
        'precond_cond': precond_cond,
        'col_cosine': col_cosine,
    }


# ============================================================
# Run experiments: sweep r, tune lr for each
# ============================================================

W1_init = torch.randn(d_hid, d_in) * 0.5
W2_init = torch.randn(d_out, d_hid) * 0.5

r_values = [2, 8, 32, 128, 512, 1024]
lr_candidates = [100.0, 10.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]

# Baseline: tune lr
print("Tuning baseline lr...")
best_base_lr, best_base_loss = None, float('inf')
for lr in lr_candidates:
    losses = train_baseline(W1_init, W2_init, lr, num_steps)
    if losses[-1] < best_base_loss:
        best_base_loss = losses[-1]
        best_base_lr = lr
print(f"  Best lr={best_base_lr}, final loss={best_base_loss:.6f}")
losses_base = train_baseline(W1_init, W2_init, best_base_lr, num_steps)

# For each r: tune lr for BC, then run diagnostics
results_bc = {}
for r in r_values:
    B1_init = torch.randn(d_hid, r)
    B2_init = torch.randn(d_out, r)
    best_lr, best_loss = None, float('inf')
    print(f"Tuning lr for BC r={r}...")
    for lr in lr_candidates:
        losses = train_BC(W1_init, W2_init, B1_init, B2_init, lr, num_steps)
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_lr = lr
    print(f"  Best lr={best_lr}, final loss={best_loss:.6f}")
    losses = train_BC(W1_init, W2_init, B1_init, B2_init, best_lr, num_steps)
    diag = get_final_diagnostics(W1_init, W2_init, B1_init, B2_init, best_lr, num_steps)
    results_bc[r] = (losses, diag, best_lr)

# For each r: tune lr for BC+DE
results_bcde = {}
for r in r_values:
    B1_init = torch.randn(d_hid, r)
    B2_init = torch.randn(d_out, r)
    D1_init = torch.randn(d_hid, k)
    D2_init = torch.randn(d_out, k)
    best_lr, best_loss = None, float('inf')
    print(f"Tuning lr for BC+DE r={r}, k={k}...")
    for lr in lr_candidates:
        losses = train_BC_DE(W1_init, W2_init, B1_init, B2_init, D1_init, D2_init, lr, num_steps)
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_lr = lr
    print(f"  Best lr={best_lr}, final loss={best_loss:.6f}")
    losses = train_BC_DE(W1_init, W2_init, B1_init, B2_init, D1_init, D2_init, best_lr, num_steps)
    results_bcde[r] = (losses, best_lr)


# ============================================================
# Plot results
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
r_labels = [str(r) for r in r_values]
x_pos = np.arange(len(r_values))

# --- Plot 1: Loss curves across r (BC only) ---
ax = axes[0, 0]
ax.semilogy(losses_base, 'k-', linewidth=2, label=f'Baseline (lr={best_base_lr})')
for r in r_values:
    losses, diag, lr = results_bc[r]
    ax.semilogy(losses, alpha=0.8, label=f'r={r} (lr={lr})')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('W+BC: Loss vs Rank r')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Plot 2: Loss curves BC vs BC+DE ---
ax = axes[0, 1]
ax.semilogy(losses_base, 'k-', linewidth=2, label=f'Baseline')
for r in r_values:
    ax.semilogy(results_bc[r][0], '-', alpha=0.7, label=f'BC r={r}')
    ax.semilogy(results_bcde[r][0], '--', alpha=0.7, label=f'BC+DE r={r},k={k}')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'BC vs BC+DE (k={k})')
ax.legend(fontsize=5)
ax.grid(True, alpha=0.3)

# --- Plot 3: Final loss vs r (BC and BC+DE) ---
ax = axes[0, 2]
final_bc = [results_bc[r][0][-1] for r in r_values]
final_bcde = [results_bcde[r][0][-1] for r in r_values]
width = 0.35
ax.bar(x_pos - width/2, final_bc, width, label='W+BC')
ax.bar(x_pos + width/2, final_bcde, width, label=f'W+BC+DE (k={k})')
ax.axhline(losses_base[-1], color='k', linestyle='--', label='Baseline')
ax.set_xticks(x_pos)
ax.set_xticklabels(r_labels)
ax.set_xlabel('Rank r')
ax.set_ylabel('Final Loss')
ax.set_yscale('log')
ax.set_title('Final Loss vs Rank')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 4: Effective rank of BB^T (bar) ---
ax = axes[1, 0]
eff_ranks = [results_bc[r][1]['effective_rank'] for r in r_values]
ax.bar(x_pos, eff_ranks, color='tab:blue')
for i, r in enumerate(r_values):
    ax.plot(i, r, 'kx', markersize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(r_labels)
ax.set_xlabel('Rank r')
ax.set_ylabel('Effective rank of B1 B1^T')
ax.set_title('Final Effective Rank (x = nominal r)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 5: BC alignment with gradient subspaces (bar) ---
ax = axes[1, 1]
align_top = [results_bc[r][1]['alignment_top'] for r in r_values]
align_bot = [results_bc[r][1]['alignment_bot'] for r in r_values]
ax.bar(x_pos - width/2, align_top, width, label='Top SVs of G', color='tab:red')
ax.bar(x_pos + width/2, align_bot, width, label='Bottom SVs of G', color='tab:blue')
ax.set_xticks(x_pos)
ax.set_xticklabels(r_labels)
ax.set_xlabel('Rank r')
ax.set_ylabel('Fractional alignment')
ax.set_title('Final B1C1 Alignment with dL/dW1 Subspaces')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 6: Column collapse (bar) ---
ax = axes[1, 2]
col_cos = [results_bc[r][1]['col_cosine'] for r in r_values]
ax.bar(x_pos, col_cos, color='tab:orange')
ax.set_xticks(x_pos)
ax.set_xticklabels(r_labels)
ax.set_xlabel('Rank r')
ax.set_ylabel('Avg |cosine| between B1 columns')
ax.set_title('Final Column Collapse of B1')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('linear_trainable_B_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary table
# ============================================================

print("\n" + "=" * 80)
print(f"{'Condition':<30s} {'Best LR':>8s} {'Final Loss':>12s} {'Speedup':>10s}")
print("=" * 80)
print(f"{'Baseline':<30s} {best_base_lr:>8.4f} {losses_base[-1]:>12.6f} {'1.00x':>10s}")
for r in r_values:
    losses, diag, lr = results_bc[r]
    target = losses_base[-1]
    steps_to_target = num_steps
    for i, l in enumerate(losses):
        if l <= target:
            steps_to_target = i
            break
    speedup = num_steps / steps_to_target if steps_to_target > 0 else float('inf')
    print(f"{'BC r=' + str(r):<30s} {lr:>8.4f} {losses[-1]:>12.6f} {speedup:>9.1f}x")
for r in r_values:
    losses, lr = results_bcde[r]
    target = losses_base[-1]
    steps_to_target = num_steps
    for i, l in enumerate(losses):
        if l <= target:
            steps_to_target = i
            break
    speedup = num_steps / steps_to_target if steps_to_target > 0 else float('inf')
    print(f"{'BC+DE r=' + str(r) + ',k=' + str(k):<30s} {lr:>8.4f} {losses[-1]:>12.6f} {speedup:>9.1f}x")
print("=" * 80)
