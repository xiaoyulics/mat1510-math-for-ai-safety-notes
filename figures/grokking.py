#!/usr/bin/env python3
"""Grokking experiment behind Figure 5.x of the Lecture 5 notes.

Run from the project root:   python3 figures/grokking.py        (needs numpy and PyTorch; about 3 minutes on a laptop CPU)
Output: figures/data/grokking_curves.dat, figures/data/grokking_neurons.dat

Task.     a + b mod p with p = 97, inputs encoded as the concatenation of two one-hot vectors (dimension 2p), 25 % of
          the p^2 pairs used for training, the rest held out.
Network.  one hidden layer without biases and with *square* activation (the architecture of Rubin-Seroussi-Ringel,
          Lecture 5):   f(x) = A (W x)^2,   W in R^{H x 2p},  A in R^{p x H},  H = 256;  cross-entropy loss
          (= negative log-likelihood of softmax probabilities, as in the lecture).
Training. full-batch AdamW, lr 1e-3, decoupled weight decay 1.0 (loosely the Gaussian prior of the lecture: the decay is decoupled from the gradient, unlike an L2 penalty),
          betas (0.9, 0.98), 10 000 steps; initialization uniform with twice the PyTorch default width.
Analysis. discrete Fourier transform of the first-layer weights: for every hidden unit i the part u_i(m) acting on the first
          number and the part v_i(n) acting on the second; the dominant frequency of each part, early (the first step at which
          the training set is fitted) and at the end, and the fraction of units for which u_i and v_i have the *same*
          dominant frequency (the claim at the end of the lecture).
"""
import math
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)

P, FRAC, H, LR, WD, INIT, STEPS = 97, 0.25, 256, 1e-3, 1.0, 2.0, 10000


def make_data(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    y = (a + b) % P
    perm = torch.randperm(P * P, generator=g)
    ntr = int(FRAC * P * P)
    X = torch.zeros(P * P, 2 * P)
    X[torch.arange(P * P), a] = 1.0
    X[torch.arange(P * P), P + b] = 1.0
    return g, X, y, perm[:ntr], perm[ntr:]


def run(seed, log_every=20, snapshot=True):
    g, X, y, tr, te = make_data(seed)
    W1 = torch.nn.Parameter(INIT * (torch.rand(H, 2 * P, generator=g) * 2 - 1) / math.sqrt(2 * P))
    W2 = torch.nn.Parameter(INIT * (torch.rand(P, H, generator=g) * 2 - 1) / math.sqrt(H))
    opt = torch.optim.AdamW([W1, W2], lr=LR, weight_decay=WD, betas=(0.9, 0.98))
    hist, early, t_train, t_test = [], None, None, None
    for step in range(STEPS + 1):
        logits = ((X[tr] @ W1.T) ** 2) @ W2.T
        loss = torch.nn.functional.cross_entropy(logits, y[tr])
        if step % log_every == 0:
            with torch.no_grad():
                lt = ((X[te] @ W1.T) ** 2) @ W2.T
                te_loss = torch.nn.functional.cross_entropy(lt, y[te]).item()
                tr_acc = (logits.argmax(1) == y[tr]).float().mean().item()
                te_acc = (lt.argmax(1) == y[te]).float().mean().item()
            hist.append((step, tr_acc, te_acc, loss.item(), te_loss, (W1.pow(2).sum() + W2.pow(2).sum()).sqrt().item()))
            if t_train is None and tr_acc >= 0.99:
                t_train = step
                early = W1.detach().clone().numpy() if snapshot else None
            if t_test is None and te_acc >= 0.99:
                t_test = step
        opt.zero_grad()
        loss.backward()
        opt.step()
    return np.array(hist), early, W1.detach().numpy(), t_train, t_test


def fourier_stats(W1):
    """Per hidden unit: dominant frequency of u_i and of v_i (folded to k = 1..(P-1)/2) and the share of the unit's
    power that sits at that frequency."""
    u, v = W1[:, :P], W1[:, P:]
    Fu, Fv = np.abs(np.fft.fft(u, axis=1)) ** 2, np.abs(np.fft.fft(v, axis=1)) ** 2
    half = (P - 1) // 2
    # fold k and P-k (real weights have conjugate-symmetric spectra); drop k = 0 (the mean)
    fold = lambda F: F[:, 1:half + 1] + F[:, P - 1:P - half - 1:-1]
    fu, fv = fold(Fu), fold(Fv)
    ku, kv = fu.argmax(1) + 1, fv.argmax(1) + 1
    pu = fu.max(1) / fu.sum(1)
    pv = fv.max(1) / fv.sum(1)
    return ku, kv, pu, pv


if __name__ == "__main__":
    t0 = time.time()
    hist, early, final, t_train, t_test = run(seed=0)
    np.savetxt(os.path.join(OUT, "grokking_curves.dat"), hist, header="step train_acc test_acc train_loss test_loss wnorm",
               comments="", fmt="%.8e")
    print(f"seed 0: training accuracy >= 99% at step {t_train}, test accuracy >= 99% at step {t_test}  ({time.time() - t0:.0f}s)")
    ku_e, kv_e, pu_e, pv_e = fourier_stats(early)
    ku_l, kv_l, pu_l, pv_l = fourier_stats(final)
    np.savetxt(os.path.join(OUT, "grokking_neurons.dat"), np.column_stack([ku_e, kv_e, ku_l, kv_l]),
               header="early_ku early_kv final_ku final_kv", comments="", fmt="%d")
    half = (P - 1) // 2
    print(f"   hidden units whose u and v have the same dominant frequency: early {np.mean(ku_e == kv_e):.3f}, "
          f"final {np.mean(ku_l == kv_l):.3f}   (chance {1 / half:.3f})")
    print(f"   median share of a unit's power at its dominant frequency (u part / v part): early {np.median(pu_e):.2f} / "
          f"{np.median(pv_e):.2f},  final {np.median(pu_l):.2f} / {np.median(pv_l):.2f}   (a flat spectrum would give {1 / half:.2f})")
    print(f"   distinct dominant frequencies among the {H} units: early {len(set(ku_e))}, final {len(set(ku_l))} of {half}")
    # robustness: other seeds
    for sd in (1, 2, 3):
        h, _, fin, tt, ts = run(seed=sd, log_every=20, snapshot=False)
        print(f"seed {sd}: training >= 99% at step {tt}, test >= 99% at step {ts}")
