"""
Compare W+BC vs Muon vs Adam vs SGD on ill-conditioned linear regression.

LR is tuned per method via grid search (short probe run).
Each rank config is plotted as a separate curve.

Usage: python test/muon.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time

torch.manual_seed(42)

# ============================================================
# Problem setup
# ============================================================

m = 128       # output dim
n = 128       # input dim
num_samples = 1024
num_steps = 5000

cond_numbers = [10, 100, 1000]
rank_configs = [[4], [8], [16], [4, 4], [8, 8]]


def make_problem(m, n, num_samples, cond_number):
    singular_values = torch.logspace(0, -np.log10(cond_number), n)
    U_data = torch.linalg.qr(torch.randn(n, n))[0]
    sqrt_cov = U_data @ torch.diag(singular_values)
    X = torch.randn(num_samples, n) @ sqrt_cov.T
    W_star = torch.randn(m, n) * 0.1
    Y = X @ W_star.T
    return X, Y, W_star


def compute_loss(W_eff, X, Y):
    return 0.5 * ((X @ W_eff.T - Y) ** 2).mean()


def get_gradient(W_eff, X, Y):
    residual = X @ W_eff.T - Y
    return residual.T @ X / X.shape[0]


def orthogonalize(M):
    X = M.float()
    X = X / (X.norm() + 1e-12)
    for _ in range(5):
        A = X @ X.T
        B = -4.7750 * A + 2.0315 * A @ A
        X = 3.4445 * X + B @ X
    return X.to(M.dtype)


def rank_label(ranks):
    if len(ranks) == 1:
        return f"r={ranks[0]}"
    return f"r={'+'.join(str(r) for r in ranks)}"


# ============================================================
# Training methods
# ============================================================

def train_sgd(W_init, X, Y, lr, num_steps):
    W = W_init.clone()
    losses = []
    for step in range(num_steps):
        loss = compute_loss(W, X, Y)
        losses.append(loss.item())
        G = get_gradient(W, X, Y)
        W = W - lr * G
    return losses


def train_adam(W_init, X, Y, lr, num_steps, beta1=0.9, beta2=0.999, eps=1e-8):
    W = W_init.clone()
    m_buf = torch.zeros_like(W)
    v_buf = torch.zeros_like(W)
    losses = []
    for step in range(num_steps):
        loss = compute_loss(W, X, Y)
        losses.append(loss.item())
        G = get_gradient(W, X, Y)
        m_buf = beta1 * m_buf + (1 - beta1) * G
        v_buf = beta2 * v_buf + (1 - beta2) * G ** 2
        m_hat = m_buf / (1 - beta1 ** (step + 1))
        v_hat = v_buf / (1 - beta2 ** (step + 1))
        W = W - lr * m_hat / (v_hat.sqrt() + eps)
    return losses


def train_muon(W_init, X, Y, lr, num_steps, momentum=0.9):
    W = W_init.clone()
    M = torch.zeros_like(W)
    losses = []
    for step in range(num_steps):
        loss = compute_loss(W, X, Y)
        losses.append(loss.item())
        G = get_gradient(W, X, Y)
        M = momentum * M + G
        G_nesterov = G + momentum * M
        update = orthogonalize(G_nesterov)
        W = W - lr * update
    return losses


# --- BC helpers ---

def _init_bc(W_init, B_inits, ranks, scale=1.0):
    W = W_init.clone()
    factors = []
    for B_init, r in zip(B_inits, ranks):
        B = B_init.clone() * scale
        C = torch.zeros(r, W_init.shape[1])
        factors.append((B, C))
    return W, factors


def _effective_weight(W, factors):
    W_eff = W
    for B, C in factors:
        W_eff = W_eff + B @ C
    return W_eff


def _bc_gradients(G, factors):
    grads = []
    for B, C in factors:
        grads.append((G @ C.T, B.T @ G))
    return grads


def _adam_step(param, grad, m_buf, v_buf, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
    m_buf.mul_(beta1).add_(grad, alpha=1 - beta1)
    v_buf.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    m_hat = m_buf / (1 - beta1 ** t)
    v_hat = v_buf / (1 - beta2 ** t)
    return param - lr * m_hat / (v_hat.sqrt() + eps)


# --- BC training functions ---

def train_bc_sgd(W_init, B_inits, X, Y, lr, num_steps, ranks=None, scale=1.0):
    W, factors = _init_bc(W_init, B_inits, ranks, scale)
    losses = []
    for step in range(num_steps):
        W_eff = _effective_weight(W, factors)
        losses.append(compute_loss(W_eff, X, Y).item())
        G = get_gradient(W_eff, X, Y)
        grads = _bc_gradients(G, factors)
        W = W - lr * G
        for i, (gB, gC) in enumerate(grads):
            B, C = factors[i]
            factors[i] = (B - lr * gB, C - lr * gC)
    return losses


def train_bc_adam(W_init, B_inits, X, Y, lr, num_steps, ranks=None, scale=1.0):
    W, factors = _init_bc(W_init, B_inits, ranks, scale)
    mW, vW = torch.zeros_like(W), torch.zeros_like(W)
    adam_bc = [((torch.zeros_like(B), torch.zeros_like(B)),
                (torch.zeros_like(C), torch.zeros_like(C)))
               for B, C in factors]
    losses = []
    for step in range(num_steps):
        W_eff = _effective_weight(W, factors)
        losses.append(compute_loss(W_eff, X, Y).item())
        G = get_gradient(W_eff, X, Y)
        grads = _bc_gradients(G, factors)
        t = step + 1
        W = _adam_step(W, G, mW, vW, lr, t)
        for i, (gB, gC) in enumerate(grads):
            B, C = factors[i]
            (mB, vB), (mC, vC) = adam_bc[i]
            B = _adam_step(B, gB, mB, vB, lr, t)
            C = _adam_step(C, gC, mC, vC, lr, t)
            factors[i] = (B, C)
            adam_bc[i] = ((mB, vB), (mC, vC))
    return losses


def train_bc_muon(W_init, B_inits, X, Y, lr, num_steps, ranks=None, scale=1.0, momentum=0.9):
    W, factors = _init_bc(W_init, B_inits, ranks, scale)
    Mom = torch.zeros_like(W)
    adam_bc = [((torch.zeros_like(B), torch.zeros_like(B)),
                (torch.zeros_like(C), torch.zeros_like(C)))
               for B, C in factors]
    losses = []
    for step in range(num_steps):
        W_eff = _effective_weight(W, factors)
        losses.append(compute_loss(W_eff, X, Y).item())
        G = get_gradient(W_eff, X, Y)
        grads = _bc_gradients(G, factors)
        Mom = momentum * Mom + G
        G_nesterov = G + momentum * Mom
        W = W - lr * orthogonalize(G_nesterov)
        t = step + 1
        for i, (gB, gC) in enumerate(grads):
            B, C = factors[i]
            (mB, vB), (mC, vC) = adam_bc[i]
            B = _adam_step(B, gB, mB, vB, lr, t)
            C = _adam_step(C, gC, mC, vC, lr, t)
            factors[i] = (B, C)
            adam_bc[i] = ((mB, vB), (mC, vC))
    return losses


# ============================================================
# Run experiments
# ============================================================

# Fixed learning rates
fixed_lrs = {
    'SGD':  0.01,
    'Adam': 0.001,
    'Muon': 0.005,
    'SGD+BC':  0.01,
    'Adam+BC': 0.001,
    'Muon+BC': 0.005,
}

base_methods = [
    ('SGD',  train_sgd,  'black',    '-'),
    ('Adam', train_adam,  'tab:blue', '-'),
    ('Muon', train_muon,  'tab:red',  '-'),
]

bc_configs = [
    ('SGD+BC',  train_bc_sgd,  'tab:orange'),
    ('Adam+BC', train_bc_adam,  'tab:purple'),
    ('Muon+BC', train_bc_muon,  'tab:green'),
]
bc_linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

fig, axes = plt.subplots(2, len(cond_numbers), figsize=(6 * len(cond_numbers), 10))
if len(cond_numbers) == 1:
    axes = axes.reshape(-1, 1)

for col, cond in enumerate(cond_numbers):
    print(f"\n{'='*60}")
    print(f"Condition number: {cond}")
    print(f"{'='*60}")

    X, Y, W_star = make_problem(m, n, num_samples, cond)
    W_init = torch.randn(m, n) * 0.5

    all_results = []  # (name, losses, lr, color, linestyle)

    # --- Base optimizers ---
    for name, train_fn, color, ls in base_methods:
        lr = fixed_lrs[name]
        t0 = time.time()
        losses = train_fn(W_init, X, Y, lr=lr, num_steps=num_steps)
        elapsed = time.time() - t0
        print(f"  {name}: lr={lr:.4e}  loss={losses[-1]:.6e}  ({elapsed:.1f}s)")
        all_results.append((name, losses, lr, color, ls))

    # --- BC variants: each rank config is a separate curve ---
    for bc_name, train_fn, color in bc_configs:
        lr = fixed_lrs[bc_name]
        for ri, ranks in enumerate(rank_configs):
            rl = rank_label(ranks)
            label = f"{bc_name} {rl}"
            B_inits = [torch.randn(m, r) for r in ranks]
            t0 = time.time()
            losses = train_fn(W_init, B_inits, X, Y, lr=lr, num_steps=num_steps, ranks=ranks)
            elapsed = time.time() - t0
            final = losses[-1]
            status = f"{final:.6e}" if np.isfinite(final) else "diverged"
            print(f"  {label}: lr={lr:.4e}  loss={status}  ({elapsed:.1f}s)")
            ls = bc_linestyles[ri % len(bc_linestyles)]
            all_results.append((label, losses, lr, color, ls))

    # --- Plot ---
    for row, (end, suffix) in enumerate([(num_steps, ''), (500, ' (first 500)')]):
        ax = axes[row, col]
        for name, losses, lr, color, ls in all_results:
            if not np.isfinite(losses[-1]):
                continue
            ax.semilogy(losses[:end], label=f"{name} (lr={lr:.1e})",
                        color=color, linestyle=ls, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'κ = {cond}{suffix}')
        ax.legend(fontsize=5)
        ax.grid(True, alpha=0.3)

    # --- Summary ---
    print(f"\n  Final losses (κ={cond}):")
    for name, losses, lr, _, _ in all_results:
        print(f"    {name:20s}  {losses[-1]:.6e}  lr={lr:.1e}")

    sgd_final = all_results[0][1][-1]
    print(f"\n  Steps to reach SGD final loss ({sgd_final:.6e}):")
    for name, losses, _, _, _ in all_results:
        steps = num_steps
        for i, l in enumerate(losses):
            if l <= sgd_final:
                steps = i
                break
        print(f"    {name:20s}  {steps:5d} steps  ({num_steps/max(steps,1):.1f}x)")

plt.tight_layout()
plt.savefig('compare_muon_bc.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n\nNote: This is a LINEAR model — no dead neurons.")
print("Your method's main advantage (revival) is NOT tested here.")
print("This comparison only measures the preconditioning effect.")
