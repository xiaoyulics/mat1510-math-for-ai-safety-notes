#!/usr/bin/env python3
"""Numerical checks and figure data for Lecture 6 (how a language model is built and trained).

Run from anywhere:   /opt/anaconda3/bin/python3 -I figures/gen_data_L6.py
Needs numpy and scipy only.  Output: figures/data/L6_*.dat  (whitespace-separated tables read by pgfplots).
Every statement of the chapter that is marked as checked by computer is an `assert` below.

Groups
------
1. Architecture (numpy transformer): causality, KV-cache identity, attention is linear only for FIXED weights,
   RoPE scores depend on relative positions only, parameter count 12 L d^2 + V d, logit vs. angle.
2. Softmax with temperature: entropy is non-decreasing in tau (dH/dbeta = -beta Var), limits, Gibbs variational principle;
   data for the temperature figure.
3. Bradley-Terry / KL-regularized RL / DPO on finite examples: Gumbel noise gives the logistic law, reward identified up
   to a constant, existence of the MLE iff the comparison graph is strongly connected, closed form of the KL-regularized
   maximizer vs. a generic optimizer, DPO = (BT MLE + closed form), consistency of the reward estimate; figure data.
4. Arithmetic quoted in the text and in the exercises (GPT-3, Chinchilla, DeepSeek-V3, Epoch data stock, KV cache,
   bits per byte, permutations).
5. Prediction = compression: adaptive character-level model (Witten-Bell interpolated orders 0..6, Bayes mixture over the
   maximal order) with a real arithmetic coder and decoder, versus zlib / bzip2 / xz, on the transcript of Lecture 6, and
   its improvement when it first sees the transcripts of other lectures ("pre-training").
"""
import bz2
import itertools
import lzma
import math
import os
import re
import time
import zlib

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
LN2 = math.log(2.0)


def save(name, header, cols):
    arr = np.column_stack(cols)
    path = os.path.join(OUT, name)
    np.savetxt(path, arr, header=header, comments="", fmt="%.8e")
    print(f"wrote {path}  shape={arr.shape}")


def softmax(z, tau=1.0):
    z = np.asarray(z, dtype=float) / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    p = np.asarray(p)
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def kl(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


# ==============================================================================================
# 1. architecture checks
# ==============================================================================================
class TinyGPT:
    """Decoder-only transformer in numpy: RMS-normalized pre-norm blocks, multi-head causal attention with RoPE, MLP
    with d_ff = 4 d (GELU), untied unembedding.  No biases and no learned gains, so that the parameter count is exactly
    12 L d^2 + 2 V d."""

    def __init__(self, V=40, d=16, L=3, H=4, rng=None, scale=0.3):
        rng = rng or np.random.default_rng(0)
        self.V, self.d, self.L, self.H = V, d, L, H
        self.dk = d // H
        s = scale / math.sqrt(d)
        self.E = rng.normal(0, 1, (V, d)) * 0.5
        self.U = rng.normal(0, 1, (V, d)) * 0.5
        self.layers = []
        for _ in range(L):
            self.layers.append(dict(
                Wq=rng.normal(0, s * 3, (d, d)), Wk=rng.normal(0, s * 3, (d, d)), Wv=rng.normal(0, s * 3, (d, d)),
                Wo=rng.normal(0, s * 3, (d, d)), W1=rng.normal(0, s * 3, (d, 4 * d)), W2=rng.normal(0, s * 3, (4 * d, d))))

    def n_params(self):
        n = self.E.size + self.U.size
        for ly in self.layers:
            n += sum(w.size for w in ly.values())
        return n

    @staticmethod
    def norm(x):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    def rope(self, x, pos):  # x: (n, H, dk)
        half = self.dk // 2
        freqs = 10000.0 ** (-np.arange(half) / half)
        ang = pos[:, None] * freqs[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    def attn_weights(self, q, k, qpos, kpos):
        """q: (nq,H,dk) k: (nk,H,dk) -> A: (H,nq,nk), causal softmax(QK^T/sqrt(dk) + mask)."""
        sc = np.einsum("qhd,khd->hqk", self.rope(q, qpos), self.rope(k, kpos)) / math.sqrt(self.dk)
        mask = kpos[None, :] > qpos[:, None]
        sc = np.where(mask[None], -np.inf, sc)
        sc = sc - sc.max(axis=-1, keepdims=True)
        e = np.exp(sc)
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, toks, return_all=False):
        n = len(toks)
        pos = np.arange(n, dtype=float)
        x = self.E[toks]
        for ly in self.layers:
            h = self.norm(x)
            q = (h @ ly["Wq"]).reshape(n, self.H, self.dk)
            k = (h @ ly["Wk"]).reshape(n, self.H, self.dk)
            v = (h @ ly["Wv"]).reshape(n, self.H, self.dk)
            A = self.attn_weights(q, k, pos, pos)
            o = np.einsum("hqk,khd->qhd", A, v).reshape(n, self.d)
            x = x + o @ ly["Wo"]
            x = x + self.gelu(self.norm(x) @ ly["W1"]) @ ly["W2"]
        logits = self.norm(x) @ self.U.T
        return logits

    def new_cache(self):
        return [dict(k=np.zeros((0, self.H, self.dk)), v=np.zeros((0, self.H, self.dk))) for _ in self.layers]

    def step(self, tok, t, cache):
        """Process ONE new token at position t using cached keys/values; returns the logits of that position."""
        x = self.E[[tok]]
        pos = np.array([float(t)])
        for ly, c in zip(self.layers, cache):
            h = self.norm(x)
            q = (h @ ly["Wq"]).reshape(1, self.H, self.dk)
            k = (h @ ly["Wk"]).reshape(1, self.H, self.dk)
            v = (h @ ly["Wv"]).reshape(1, self.H, self.dk)
            c["k"] = np.concatenate([c["k"], k], 0)
            c["v"] = np.concatenate([c["v"], v], 0)
            kpos = np.arange(c["k"].shape[0], dtype=float)
            A = self.attn_weights(q, c["k"], pos, kpos)
            o = np.einsum("hqk,khd->qhd", A, c["v"]).reshape(1, self.d)
            x = x + o @ ly["Wo"]
            x = x + self.gelu(self.norm(x) @ ly["W1"]) @ ly["W2"]
        return (self.norm(x) @ self.U.T)[0]


def architecture_checks():
    print("\n[1] architecture checks")
    rng = np.random.default_rng(1)
    net = TinyGPT(V=40, d=16, L=3, H=4, rng=rng)
    toks = rng.integers(0, net.V, 12)
    full = net.forward(toks)

    # (a) parameter count: 12 L d^2 + 2 V d  (untied embedding and unembedding, no biases / gains)
    assert net.n_params() == 12 * net.L * net.d ** 2 + 2 * net.V * net.d
    print("   parameter count = 12 L d^2 + 2 V d  ->", net.n_params())

    # (b) causality: changing token m does not change the logits at positions < m
    m = 7
    t2 = toks.copy()
    t2[m] = (t2[m] + 5) % net.V
    full2 = net.forward(t2)
    assert np.max(np.abs(full2[:m] - full[:m])) < 1e-12
    assert np.max(np.abs(full2[m:] - full[m:])) > 1e-6
    print("   causality: positions < m unchanged, positions >= m changed")

    # (c) KV cache: incremental decoding reproduces the full forward pass exactly
    cache = net.new_cache()
    inc = np.stack([net.step(int(tok), t, cache) for t, tok in enumerate(toks)])
    err = np.max(np.abs(inc - full))
    assert err < 1e-10, err
    n = len(toks)
    print(f"   KV cache identity: max |incremental - full| = {err:.2e};  positions processed: cached {n}, recompute {n*(n+1)//2}")

    # (d) attention is linear in the values for FIXED weights, not linear in the input
    d, H, dk = net.d, net.H, net.dk
    ly = net.layers[0]
    n = 6
    X1 = rng.normal(size=(n, d))
    X2 = rng.normal(size=(n, d))
    pos = np.arange(n, dtype=float)

    def attn_out(X, A=None):
        h = X
        q = (h @ ly["Wq"]).reshape(n, H, dk)
        k = (h @ ly["Wk"]).reshape(n, H, dk)
        v = (h @ ly["Wv"]).reshape(n, H, dk)
        if A is None:
            A = net.attn_weights(q, k, pos, pos)
        return np.einsum("hqk,khd->qhd", A, v).reshape(n, d) @ ly["Wo"], A

    out1, A1 = attn_out(X1)
    out_frozen_sum, _ = attn_out(X1 + X2, A=A1)          # frozen weights: linear in X
    o2, _ = attn_out(X2, A=A1)
    assert np.max(np.abs(out_frozen_sum - (out1 + o2))) < 1e-12
    out12, _ = attn_out(X1 + X2)
    o2n, _ = attn_out(X2)
    gap_add = np.max(np.abs(out12 - (out1 + o2n)))
    out_2x, _ = attn_out(2 * X1)
    gap_hom = np.max(np.abs(out_2x - 2 * out1))
    assert gap_add > 1e-2 and gap_hom > 1e-2
    print(f"   attention: linear for frozen weights (err < 1e-12); true map: additivity gap {gap_add:.3f}, homogeneity gap {gap_hom:.3f}")

    # (e) RoPE: the score q_i . k_j depends only on i - j (shift all positions by s)
    q = rng.normal(size=(5, H, dk))
    k = rng.normal(size=(5, H, dk))
    p = np.arange(5, dtype=float)
    s1 = np.einsum("qhd,khd->hqk", net.rope(q, p), net.rope(k, p))
    s2 = np.einsum("qhd,khd->hqk", net.rope(q, p + 37), net.rope(k, p + 37))
    assert np.max(np.abs(s1 - s2)) < 1e-10
    print("   RoPE: scores invariant under a global shift of positions (depend on i-j only)")

    # (f) logit = <h,u> + b, not an angle: a token whose vector has the smallest angle need not be the most probable
    h = np.array([1.0, 0.0])
    U = np.array([[0.5, 0.2], [2.0, 2.0]])          # token 0 has smaller angle to h, token 1 has the larger inner product
    ang = np.degrees(np.arccos(U @ h / (np.linalg.norm(U, axis=1) * np.linalg.norm(h))))
    logit = U @ h
    assert ang[0] < ang[1] and logit[1] > logit[0]
    print("   angle example: angles (deg) =", np.round(ang, 2), " logits =", logit, " softmax =", np.round(softmax(logit), 4))
    # with equal norms the order of the probabilities is the order of the cosines
    U2 = rng.normal(size=(9, 5))
    U2 /= np.linalg.norm(U2, axis=1, keepdims=True)
    h2 = rng.normal(size=5)
    cosv = U2 @ h2 / np.linalg.norm(h2)
    assert np.all(np.argsort(softmax(U2 @ h2)) == np.argsort(cosv))
    # ... and the norm of h acts as an inverse temperature: p ~ exp(|h| cos(theta)/tau)
    assert np.allclose(softmax(U2 @ (3 * h2)), softmax(cosv * 3 * np.linalg.norm(h2)))


# ==============================================================================================
# 2. temperature
# ==============================================================================================
LOGITS = np.array([4.0, 2.6, 1.9, 1.3, 0.6, 0.2, -0.3, -1.0])
TOK = ["Paris", "the", "a", "located", "Lyon", "known", "France", "Rome"]


def temperature_checks():
    print("\n[2] temperature")
    z = LOGITS
    # entropy non-decreasing in tau, derivative formula dH/dbeta = -beta Var_beta(z)
    taus = np.logspace(-1.6, 1.6, 400)
    H = np.array([entropy(softmax(z, t)) for t in taus])
    assert np.all(np.diff(H) > 0)
    for beta in [0.3, 1.0, 2.5]:
        eps = 1e-6
        Hb = lambda b: entropy(softmax(z * b))
        dnum = (Hb(beta + eps) - Hb(beta - eps)) / (2 * eps)
        p = softmax(z * beta)
        var = float((p * z ** 2).sum() - (p * z).sum() ** 2)
        assert abs(dnum - (-beta * var)) < 1e-7, (dnum, -beta * var)
    # limits
    assert np.allclose(softmax(z, 1e-3), np.eye(len(z))[0], atol=1e-9)
    assert np.max(np.abs(softmax(z, 1e4) - 1.0 / len(z))) < 1e-3
    assert abs(entropy(softmax(z, 1e5)) - math.log(len(z))) < 1e-6
    # ties: tau -> 0 gives the uniform distribution on the argmax set
    zt = np.array([2.0, 2.0, 1.0])
    assert np.allclose(softmax(zt, 1e-4), [0.5, 0.5, 0.0], atol=1e-9)
    # Gibbs variational principle: <pi, z> + tau H(pi) <= tau log Z, equality at pi = softmax(z/tau)
    rng = np.random.default_rng(3)
    for tau in [0.4, 1.0, 2.0]:
        logZ = tau * logsumexp(z / tau)
        p = softmax(z, tau)
        assert abs(p @ z + tau * entropy(p) - logZ) < 1e-12
        for _ in range(500):
            q = rng.dirichlet(np.ones(len(z)) * rng.uniform(0.2, 3))
            assert q @ z + tau * entropy(q) <= logZ + 1e-12
    print("   entropy increasing in tau; dH/dbeta = -beta Var; limits; Gibbs variational principle: OK")

    # data for the figure
    tau3 = [0.5, 1.0, 2.0]
    cols = [np.arange(len(z))] + [softmax(z, t) for t in tau3]
    save("L6_temp_bars.dat", "idx p_tau05 p_tau10 p_tau20", cols)
    print("   H at tau=0.5,1,2:", [round(entropy(softmax(z, t)), 4) for t in tau3], " log V =", round(math.log(len(z)), 4))
    print("   p(Paris) at tau=0.5,1,2:", [round(float(softmax(z, t)[0]), 4) for t in tau3])
    save("L6_temp_entropy.dat", "tau H Hbits", [taus, H, H / LN2])
    return tau3


# ==============================================================================================
# 3. Bradley-Terry, KL-regularized optimum, DPO
# ==============================================================================================
def bt_nll(r, w, l):
    """mean of -log sigma(r_w - r_l)"""
    return float(np.mean(np.logaddexp(0.0, -(r[w] - r[l]))))


def bt_fit(K, w, l, ridge=0.0):
    def f(r):
        d = r[w] - r[l]
        val = np.mean(np.logaddexp(0.0, -d)) + 0.5 * ridge * np.sum(r * r)
        g = np.zeros(K)
        c = -expit(-d) / len(w)
        np.add.at(g, w, c)
        np.add.at(g, l, -c)
        return val, g + ridge * r
    res = minimize(f, np.zeros(K), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-11))
    return res.x - res.x.mean()


def strongly_connected(K, w, l):
    A = np.zeros((K, K))
    A[w, l] = 1.0
    ncomp, _ = connected_components(A, directed=True, connection="strong")
    return ncomp == 1


def closed_form_policy(pref, r, beta):
    lp = np.log(pref) + r / beta
    return np.exp(lp - logsumexp(lp))


def bt_checks():
    print("\n[3] Bradley-Terry / KL / DPO")
    rng = np.random.default_rng(5)

    # (a) Gumbel noise => logistic law:  P(r_w + g_w > r_l + g_l) = sigma(r_w - r_l)
    for delta in [0.0, 1.0, 2.0, 4.6]:
        n = 4_000_000
        g = rng.gumbel(size=(2, n))
        freq = np.mean(delta + g[0] > g[1])
        se = math.sqrt(0.25 / n)
        assert abs(freq - expit(delta)) < 5 * se, (delta, freq, expit(delta))
    print("   sigma(delta):", {dl: round(float(expit(dl)), 4) for dl in [0.0, 1.0, 2.0, 3.0, 4.6]})

    # (b) identifiability up to an additive constant
    K = 6
    r_true = rng.normal(size=K)
    w = rng.integers(0, K, 200)
    l = (w + rng.integers(1, K, 200)) % K
    assert abs(bt_nll(r_true, w, l) - bt_nll(r_true + 3.7, w, l)) < 1e-12
    # but not under non-constant shifts
    assert abs(bt_nll(r_true, w, l) - bt_nll(r_true + np.arange(K) * 0.3, w, l)) > 1e-6

    # (c) MLE exists iff the comparison graph is strongly connected
    #    graph 1: 0>1, 1>2, 2>0 (cycle) plus some reversals -> strongly connected
    w1 = np.array([0, 1, 2, 0, 1, 0, 0, 1]); l1 = np.array([1, 2, 0, 1, 2, 2, 1, 0])
    assert strongly_connected(3, w1, l1)
    r1 = bt_fit(3, w1, l1)
    f1 = bt_nll(r1, w1, l1)
    for scale in [1.5, 3.0]:
        assert bt_nll(r1 * scale, w1, l1) > f1 - 1e-12        # the minimum is attained at the fitted r
    #    graph 2: 0 always beats 1, 1 always beats 2 (transitive, no reversal) -> not strongly connected
    w2 = np.array([0, 0, 1, 1, 0]); l2 = np.array([1, 2, 2, 2, 1])
    assert not strongly_connected(3, w2, l2)
    vals = [bt_nll(np.array([2.0, 0.0, -2.0]) * s, w2, l2) for s in [1, 2, 4, 8, 16]]
    assert all(vals[i + 1] < vals[i] for i in range(len(vals) - 1)) and vals[-1] < 1e-6
    print("   MLE exists iff strongly connected: cycle example fitted (r =", np.round(r1, 3), "); transitive example: loss ->", f"{vals[-1]:.1e}", "along a ray")

    # (d) consistency, with the asymptotic law  E|r_hat - r|^2 ~ tr(I^+) / n
    K = 8
    r_true = np.sort(rng.normal(size=K))
    r_true -= r_true.mean()
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    # Fisher information per comparison (pair chosen uniformly): I = mean_{pairs} p(1-p) (e_i-e_j)(e_i-e_j)^T
    I = np.zeros((K, K))
    for i, j in pairs:
        p = expit(r_true[i] - r_true[j])
        e = np.zeros(K); e[i], e[j] = 1, -1
        I += p * (1 - p) * np.outer(e, e)
    I /= len(pairs)
    tr_pinv = float(np.trace(np.linalg.pinv(I)))
    ns = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    reps = 60
    rmse = []
    for n in ns:
        errs = []
        for _ in range(reps):
            while True:
                idx = rng.integers(0, len(pairs), n)
                a = np.array([pairs[t][0] for t in idx]); b = np.array([pairs[t][1] for t in idx])
                win_a = rng.random(n) < expit(r_true[a] - r_true[b])
                w = np.where(win_a, a, b); l = np.where(win_a, b, a)
                if strongly_connected(K, w, l):
                    break
            rh = bt_fit(K, w, l)
            errs.append(np.mean((rh - r_true) ** 2))
        rmse.append(math.sqrt(np.mean(errs)))
    rmse = np.array(rmse)
    pred = np.sqrt(tr_pinv / K / ns)
    slope = np.polyfit(np.log(ns[2:]), np.log(rmse[2:]), 1)[0]
    assert abs(slope + 0.5) < 0.06, slope
    assert np.all(np.abs(rmse[3:] / pred[3:] - 1) < 0.15), rmse / pred
    print(f"   consistency: log-log slope {slope:.3f} (expected -0.5); simulated/asymptotic RMSE at n=50000: {rmse[-1]/pred[-1]:.3f}")
    save("L6_bt_consistency.dat", "n rmse asym", [ns, rmse, pred])

    # (e) KL-regularized maximizer
    pref = np.array([0.45, 0.30, 0.15, 0.07, 0.03])
    r = np.array([0.0, 0.5, 1.0, 2.0, 3.0])

    def J(pi, beta):
        return float(pi @ r - beta * kl(pi, pref))

    def negJ_and_grad(th, beta):
        pi = np.exp(th - logsumexp(th))
        a = r - beta * (np.log(pi) - np.log(pref))            # dJ/dpi_i = a_i - beta ;  dJ/dtheta_j = pi_j (a_j - sum_i pi_i a_i)
        return -float(pi @ a), -(pi * (a - pi @ a))

    for beta in [0.3, 0.5, 1.0, 4.0]:
        star = closed_form_policy(pref, r, beta)
        logZ = logsumexp(np.log(pref) + r / beta)
        assert abs(J(star, beta) - beta * logZ) < 1e-12
        for _ in range(300):
            q = rng.dirichlet(np.ones(5) * rng.uniform(0.3, 4))
            assert abs(J(q, beta) - (beta * logZ - beta * kl(q, star))) < 1e-10     # J = beta log Z - beta KL(q || pi*)
        res = minimize(negJ_and_grad, np.zeros(5), args=(beta,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, ftol=1e-16, gtol=1e-12))
        num = np.exp(res.x - logsumexp(res.x))
        assert np.max(np.abs(num - star)) < 1e-6, (beta, num, star)
        # invariance under r -> r + c
        assert np.allclose(closed_form_policy(pref, r + 5.0, beta), star, atol=1e-14)
    print("   KL-regularized optimum: closed form = generic optimizer (< 1e-6); J = beta log Z - beta KL(pi||pi*); invariant under r -> r + c")

    # figure data: pi*_beta (5 responses) and the reward-KL frontier
    betas = np.logspace(-1.5, 1.5, 120)
    P = np.array([closed_form_policy(pref, r, b) for b in betas])
    Er = P @ r
    KLs = np.array([kl(p, pref) for p in P])
    save("L6_kl_policy.dat", "beta p1 p2 p3 p4 p5 Er KL", [betas] + [P[:, i] for i in range(5)] + [Er, KLs])
    assert np.all(np.diff(Er) < 0) and np.all(np.diff(KLs) < 0)       # as beta grows, reward and KL both fall
    print("   beta -> 0: pi* ->", np.round(closed_form_policy(pref, r, 1e-3), 4), " beta -> inf: pi* ->", np.round(closed_form_policy(pref, r, 1e4), 4))

    # (f) DPO = BT maximum likelihood + closed-form KL-regularized policy (tabular case)
    n = 6000
    beta = 0.5
    K = 5
    a = rng.integers(0, K, n)
    b = (a + rng.integers(1, K, n)) % K
    win_a = rng.random(n) < expit(r[a] - r[b])
    w = np.where(win_a, a, b); l = np.where(win_a, b, a)
    rhat = bt_fit(K, w, l)                                       # stage 1: reward from comparisons
    two_stage = closed_form_policy(pref, rhat, beta)             # stage 2: exact KL-regularized optimum

    def dpo_loss(th):
        lp = th - logsumexp(th)
        ratio = lp - np.log(pref)
        return float(np.mean(np.logaddexp(0.0, -beta * (ratio[w] - ratio[l]))))
    res = minimize(dpo_loss, np.zeros(K), method="BFGS", options=dict(gtol=1e-12, maxiter=2000))
    dpo_pol = np.exp(res.x - logsumexp(res.x))
    assert np.max(np.abs(dpo_pol - two_stage)) < 1e-6, (dpo_pol, two_stage)
    implicit = beta * (np.log(dpo_pol) - np.log(pref))
    assert np.max(np.abs((implicit - implicit.mean()) - rhat)) < 1e-5
    print("   DPO policy = two-stage policy (max diff %.1e); implicit reward = BT estimate up to a constant (max diff %.1e)"
          % (np.max(np.abs(dpo_pol - two_stage)), np.max(np.abs((implicit - implicit.mean()) - rhat))))
    print("   true r (centered):", np.round(r - r.mean(), 3), " r_hat:", np.round(rhat, 3))


# ==============================================================================================
# 4. arithmetic quoted in the text
# ==============================================================================================
def arithmetic():
    print("\n[4] arithmetic")
    # --- GPT-3 175B (Brown et al. 2020): L=96, d=12288, V=50257 (GPT-2 BPE), n_ctx=2048, trained on 300B tokens
    L, d, V, nctx, D = 96, 12288, 50257, 2048, 300e9
    core = 12 * L * d ** 2
    tot = core + V * d + nctx * d
    print(f"   GPT-3: 12 L d^2 = {core:.4e};  + V d + n_ctx d = {tot:.4e}  (paper: 174,600 M)")
    assert abs(tot / 1e6 - 174_600) / 174_600 < 2e-4
    C = 6 * 174.6e9 * D
    print(f"   C = 6 N D = {C:.3e}  (paper Table D.1: 3.14e23)")
    assert abs(C - 3.14e23) / 3.14e23 < 2e-3
    assert abs(410 + 19 + 12 + 55 + 3 - 499) == 0                   # Table 2.2: 499 B tokens in the mixture
    # attention-score overhead relative to the 2*12 L d^2 matmul FLOPs of the weights: 4 n d L / (24 L d^2) = n / (6 d)
    print(f"   attention overhead n/(6d) at n=2048: {nctx/(6*d):.4f}")
    # tokens per parameter
    print(f"   tokens per parameter: GPT-3 {300e9/175e9:.2f}, Chinchilla {1.4e12/70e9:.1f}, DeepSeek-V3 total {14.8e12/671e9:.1f} / active {14.8e12/37e9:.0f}")
    # --- Chinchilla: 70B, 1.4T tokens, d=8192, 80 layers; Gopher-budget 5.76e23 (their exact FLOP count)
    print(f"   Chinchilla 12 L d^2 = {12*80*8192**2:.3e};  6ND = {6*70e9*1.4e12:.3e} (paper 5.76e23 with exact count)")
    # --- DeepSeek-V3: GPU hours 2664K + 119K + 5K = 2788K
    ph = 5 / 2788
    print(f"   DeepSeek-V3 post-training share of GPU hours: {100*ph:.3f}% ; pre-training {100*2664/2788:.2f}%")
    assert 2664 + 119 + 5 == 2788
    c_v3 = 6 * 37e9 * 14.8e12
    print(f"   DeepSeek-V3 C = 6 N_active D = {c_v3:.3e} FLOP;  per GPU-hour {c_v3/2.664e6:.3e} FLOP = {c_v3/2.664e6/3600/1e12:.0f} TFLOP/s per GPU")
    # --- compute-optimal arithmetic with N = D/20:  C = 6 N D = 0.3 D^2
    for D_, label in [(15e12, "15T tokens"), (320e12, "320T (Epoch repetition-adjusted stock)"), (4e14, "4e14 (Epoch stock in 2028)")]:
        N_ = D_ / 20
        print(f"   D={label}: N = {N_:.3e}, C = {6*N_*D_:.3e}")
    assert abs(6 * (400e12 / 20) * 400e12 - 4.8e28) / 4.8e28 < 1e-12          # Epoch: ~5e28 FLOP for 4e14 tokens
    # Epoch: 5e25 FLOP, Chinchilla scaling -> ~13T tokens
    print(f"   C = 5e25 FLOP, N = D/20: D = {math.sqrt(20*5e25/6):.3e}")
    # InstructGPT: 33k prompts, K in [4, 9] ranked answers => C(K,2) in [6, 36] comparisons per prompt
    print("   InstructGPT comparisons: %d .. %d" % (33000 * math.comb(4, 2), 33000 * math.comb(9, 2)))
    # --- KV cache for GPT-3-like (fp16): 2 (K and V) * L * d * 2 bytes per token
    per_tok = 2 * L * d * 2
    print(f"   KV cache per token (GPT-3 shape, fp16): {per_tok} bytes = {per_tok/1e6:.2f} MB; 100k tokens: {per_tok*1e5/1e9:.1f} GB; 2048 tokens: {per_tok*2048/1e9:.2f} GB")
    # generation of T tokens after a prompt of P tokens: positions processed with and without cache
    P_, T_ = 1000, 500
    # to produce T new tokens after a prompt of P tokens: step t (t = 1..T) needs the states of P + t - 1 positions
    with_cache = P_ + (T_ - 1)                                   # prefill once, then one new position per step
    no_cache = sum(P_ + t - 1 for t in range(1, T_ + 1))        # recompute the whole prefix at every step
    assert no_cache == T_ * P_ + T_ * (T_ - 1) // 2 and with_cache == 1499 and no_cache == 624750
    print(f"   positions processed, P={P_}, T={T_}: with cache {with_cache}, without {no_cache}  (ratio {no_cache/with_cache:.1f})")
    # --- bits per byte from per-token loss
    loss_nats, bytes_per_token = 2.0, 4.0
    bpb = loss_nats / (bytes_per_token * LN2)
    print(f"   2.0 nats/token, 4 bytes/token: {bpb:.4f} bits per byte = {100*bpb/8:.2f}% of raw size; perplexity {math.exp(loss_nats):.2f}")
    # --- the 'prefix of the training data' example for the text: bits per byte of Chinchilla 70B on enwik9 is 8.3% of 8 bits
    print(f"   Chinchilla 70B on enwik9: 8.3% -> {0.083*8:.3f} bits/byte;  xz(LZMA2) 23.0% -> {0.23*8:.2f};  gzip 32.3% -> {0.323*8:.2f}")
    # --- S5 word problem: n = 2^k permutations composed by a balanced tree in k rounds
    rng = np.random.default_rng(7)
    k = 6
    perms = [tuple(rng.permutation(5)) for _ in range(2 ** k)]
    compose = lambda p, q: tuple(q[i] for i in p)                       # apply p first, then q
    seq = perms[0]
    for p in perms[1:]:
        seq = compose(seq, p)
    layer = perms
    rounds = 0
    while len(layer) > 1:
        layer = [compose(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
        rounds += 1
    assert layer[0] == seq and rounds == k
    print(f"   S5: sequential scratchpad needs {2**k - 1} compositions, balanced tree needs {rounds} rounds; results agree")
    # sizes of the scratchpad vs. tree: composition is associative (checked on random triples)
    for _ in range(200):
        a, b, c = (tuple(rng.permutation(5)) for _ in range(3))
        assert compose(compose(a, b), c) == compose(a, compose(b, c))
    # --- SFT / loss mask example
    lp = np.array([-2.1, -0.4, -1.3, -0.7, -0.2, -0.9])        # log-probs of 6 tokens of a toy sequence
    mask = np.array([0, 0, 1, 1, 1, 1])
    print(f"   masked SFT loss on last 4 of 6 tokens: {-(lp*mask).sum()/mask.sum():.4f} nats/token (sum {-(lp*mask).sum():.2f})")
    # --- DPO example for the exercise: log-ratios
    beta = 0.1
    lw, ll = 1.2, -0.8       # log pi_theta/pi_ref of winner and loser
    z = beta * (lw - ll)
    print(f"   DPO exercise: beta*(lw - ll) = {z:.3f},  loss = -log sigma = {float(np.logaddexp(0, -z)):.4f}")
    # --- KL-regularized exercise
    pref = np.array([0.5, 0.3, 0.2]); r = np.array([0.0, 1.0, 2.0])
    star = closed_form_policy(pref, r, 1.0)
    print("   KL exercise (pi_ref=(.5,.3,.2), r=(0,1,2), beta=1): pi* =", np.round(star, 4), " J* = beta log Z =", round(float(logsumexp(np.log(pref) + r)), 4),
          " E r =", round(float(star @ r), 4), " KL =", round(kl(star, pref), 4))
    star2 = closed_form_policy(pref, r, 0.5)
    print("   same with beta=0.5: pi* =", np.round(star2, 4), " J* =", round(0.5 * float(logsumexp(np.log(pref) + r / 0.5)), 4))
    # --- temperature exercise
    zz = np.array([3.0, 1.0, 0.0])
    for t in [0.5, 1.0, 2.0]:
        p = softmax(zz, t)
        print(f"   z=(3,1,0), tau={t}: p = {np.round(p,4)}, H = {entropy(p):.4f} nats")
    # --- BT exercise: P(w>l) for gap 1, 2 and the gap needed for 99%
    print("   BT: sigma(1) = %.4f, sigma(2) = %.4f, gap for 99%% = %.4f" % (expit(1), expit(2), math.log(99)))
    # --- BT exercise 4: A>B x3, B>C x3, A>C x3 has no maximum-likelihood reward; adding C>A x1 gives a cycle and a maximizer
    A_, B_, C_ = 0, 1, 2
    w = np.array([A_] * 3 + [B_] * 3 + [A_] * 3)
    l = np.array([B_] * 3 + [C_] * 3 + [C_] * 3)
    assert not strongly_connected(3, w, l)
    vals = [bt_nll(np.array([1.0, 0.0, -1.0]) * t, w, l) for t in [1, 2, 4, 8]]
    assert all(vals[i + 1] < vals[i] for i in range(3)) and vals[-1] < 1e-3
    w1, l1 = np.append(w, C_), np.append(l, A_)
    assert strongly_connected(3, w1, l1)
    rh = bt_fit(3, w1, l1)
    a_ = rh[0]
    assert abs(rh[1]) < 1e-6 and abs(a_ + rh[2]) < 1e-6
    assert abs(6 * expit(-a_) + 6 * expit(-2 * a_) - 2 * expit(2 * a_)) < 1e-8
    print("   BT exercise 4: loss along r = t(1,0,-1), t=1,2,4,8:", np.round(vals, 5), "; with C>A added r_hat =", np.round(rh, 4))
    # --- asserted answers of the exercises (hints in fragments/L06.tex)
    assert np.allclose(softmax(zz, 0.5), [0.9796, 0.0179, 0.0024], atol=6e-5) and abs(entropy(softmax(zz, 0.5)) - 0.1069) < 5e-5
    assert np.allclose(softmax(zz, 1.0), [0.8438, 0.1142, 0.0420], atol=6e-5) and abs(entropy(softmax(zz, 1.0)) - 0.5243) < 5e-5
    assert np.allclose(softmax(zz, 2.0), [0.6285, 0.2312, 0.1402], atol=6e-5) and abs(entropy(softmax(zz, 2.0)) - 0.9060) < 5e-5
    assert np.allclose(star, [0.1790, 0.2919, 0.5291], atol=6e-5) and abs(float(star @ r) - 1.3501) < 5e-5 and abs(kl(star, pref) - 0.3228) < 5e-5
    assert abs(float(logsumexp(np.log(pref) + r)) - 1.0272) < 5e-5 and np.allclose(star2, [0.0367, 0.1626, 0.8008], atol=6e-5)
    assert abs(0.5 * float(logsumexp(np.log(pref) + r / 0.5)) - 1.3064) < 5e-5
    assert abs(float(np.logaddexp(0, -0.2)) - 0.5981) < 5e-5
    assert abs(math.exp(2.0) - 7.389) < 1e-3 and abs(bpb - 0.7213) < 5e-5 and abs(math.log2(7) - 2.807) < 1e-3


# ==============================================================================================
# 5. prediction = compression
# ==============================================================================================
def clean_text(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    lines = [re.sub(r"^\[(?:\d+:)?\d\d:\d\d\]\s*", "", ln) for ln in raw.splitlines()]
    txt = " ".join(ln.strip() for ln in lines if ln.strip())
    return re.sub(r"\s+", " ", txt).encode("utf-8")


class CharModel:
    """Orders 0..KMAX with Witten-Bell interpolation:  p_k(x|c) = (n_c(x) + t_c p_{k-1}(x|c')) / (n_c + t_c),
    t_c = number of distinct symbols seen after context c; p_{-1} uniform on 256 byte values.  Expert K = order-K model;
    the prediction is the Bayes mixture over K with uniform prior (weights proportional to past likelihood)."""

    def __init__(self, kmax=6):
        self.kmax = kmax
        self.tab = [dict() for _ in range(kmax + 1)]       # per order: context bytes -> [total, distinct, {sym: count}]
        self.logw = np.zeros(kmax + 1)                      # log prior weights (uniform, normalized lazily)
        self.hist = bytearray()

    def _ctx(self, k):
        h = self.hist
        return bytes(h[len(h) - k:]) if k > 0 else b""

    def expert_probs(self, sym):
        """probabilities of byte `sym` under experts K = 0..kmax (one recursion)"""
        p = 1.0 / 256.0
        out = []
        for k in range(self.kmax + 1):
            if len(self.hist) < k:
                out.append(p)           # context not available: fall back to the lower order
                continue
            e = self.tab[k].get(self._ctx(k))
            if e is not None:
                n, t, d = e
                p = (d.get(sym, 0) + t * p) / (n + t)
            out.append(p)
        return out

    def expert_vectors(self):
        """full distributions (256,) of experts K = 0..kmax"""
        p = np.full(256, 1.0 / 256.0)
        out = []
        for k in range(self.kmax + 1):
            if len(self.hist) >= k:
                e = self.tab[k].get(self._ctx(k))
                if e is not None:
                    n, t, d = e
                    q = t * p
                    for s, c in d.items():
                        q[s] += c
                    p = q / (n + t)
            out.append(p)
        return out

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def update(self, sym, ps=None):
        if ps is not None:
            self.logw += np.log(np.array(ps))
        for k in range(self.kmax + 1):
            if len(self.hist) < k:
                continue
            c = self._ctx(k)
            e = self.tab[k].get(c)
            if e is None:
                self.tab[k][c] = [1, 1, {sym: 1}]
            else:
                if sym in e[2]:
                    e[2][sym] += 1
                else:
                    e[2][sym] = 1
                    e[1] += 1
                e[0] += 1
        self.hist.append(sym)

    def train(self, data):
        for s in data:
            self.update(s)

    def code_length_bits(self, data):
        """sequential code length of `data` (bits) under the mixture and under each fixed-order expert"""
        total_mix = 0.0
        tot = np.zeros(self.kmax + 1)
        for s in data:
            ps = self.expert_probs(s)
            w = self.weights()
            pm = float(np.dot(w, ps))
            total_mix += -math.log2(pm)
            tot += -np.log2(np.array(ps))
            self.update(s, ps)
        return total_mix, tot


class ArithmeticCoder:
    """Textbook (Witten-Neal-Cleary) integer arithmetic coder with Python big integers (P = 62 bits of range)."""
    P = 62
    TOP = (1 << P) - 1
    HALF = 1 << (P - 1)
    QUARTER = 1 << (P - 2)

    @staticmethod
    def quantize(p, total_bits=40):
        """integer frequencies >= 1 summing to exactly 2^total_bits, derived deterministically from p"""
        total = 1 << total_bits
        f = np.floor(p * (total - 256)).astype(np.int64) + 1
        f[np.argmax(f)] += total - int(f.sum())
        return f

    @classmethod
    def encode(cls, symbols, prob_stream):
        """prob_stream(t) -> probability vector for position t given the symbols before t (called in order)"""
        low, high, pending = 0, cls.TOP, 0
        bits = []

        def emit(b):
            nonlocal pending
            bits.append(b)
            bits.extend([1 - b] * pending)
            pending = 0

        for t, s in enumerate(symbols):
            f = cls.quantize(prob_stream(t))
            cum = int(f[:s].sum())
            tot = int(f.sum())
            rng_ = high - low + 1
            high = low + rng_ * (cum + int(f[s])) // tot - 1
            low = low + rng_ * cum // tot
            while True:
                if high < cls.HALF:
                    emit(0)
                elif low >= cls.HALF:
                    emit(1)
                    low -= cls.HALF
                    high -= cls.HALF
                elif low >= cls.QUARTER and high < 3 * cls.QUARTER:
                    pending += 1
                    low -= cls.QUARTER
                    high -= cls.QUARTER
                else:
                    break
                low = 2 * low
                high = 2 * high + 1
        pending += 1
        emit(0 if low < cls.QUARTER else 1)
        return bits

    @classmethod
    def decode(cls, bits, n, prob_stream):
        pos = [0]

        def nxt():
            i = pos[0]
            pos[0] += 1
            return bits[i] if i < len(bits) else 0

        low, high = 0, cls.TOP
        value = 0
        for _ in range(cls.P):
            value = 2 * value + nxt()
        out = []
        for t in range(n):
            f = cls.quantize(prob_stream(t, out))
            tot = int(f.sum())
            rng_ = high - low + 1
            target = ((value - low + 1) * tot - 1) // rng_
            cumf = np.cumsum(f)
            s = int(np.searchsorted(cumf, target, side="right"))
            cum = int(cumf[s] - f[s])
            high = low + rng_ * int(cumf[s]) // tot - 1
            low = low + rng_ * cum // tot
            out.append(s)
            while True:
                if high < cls.HALF:
                    pass
                elif low >= cls.HALF:
                    value -= cls.HALF
                    low -= cls.HALF
                    high -= cls.HALF
                elif low >= cls.QUARTER and high < 3 * cls.QUARTER:
                    value -= cls.QUARTER
                    low -= cls.QUARTER
                    high -= cls.QUARTER
                else:
                    break
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value + nxt()
        return out


def baseline_sizes(data):
    z = len(zlib.compress(data, 9)) - 6                         # zlib stream = 2-byte header + 4-byte Adler-32 around raw deflate
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    zraw = len(co.compress(data) + co.flush())
    b = len(bz2.compress(data, 9))
    xz = len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]))
    return dict(zlib_raw=zraw, bz2=b, xz_raw=xz)


def build_pretraining_text(tdir, names, chunk=4000, seed=11):
    """other lectures' transcripts cut into chunks and shuffled with a fixed seed, so that every prefix is a mixture of
    all of them (the guest lecture L4 and the lectures L2, L5, L7, L8)."""
    chunks = []
    for k in ["L5", "L7", "L8", "L4", "L2"]:
        t = clean_text(os.path.join(tdir, names[k]))
        chunks += [t[i:i + chunk] for i in range(0, len(t), chunk)]
    order = np.random.default_rng(seed).permutation(len(chunks))
    return b" ".join(chunks[i] for i in order)


def compression_experiment():
    print("\n[5] prediction = compression")
    tdir = os.path.join(ROOT, "transcripts")
    names = {"L6": "-cqba5tE6aQ.txt", "L2": "-XmDvvHqfLI.txt", "L4": "7AaNkZuYfmk.txt", "L5": "xRYdgd_G-j0.txt",
             "L7": "CQ_ECK2lr9g.txt", "L8": "jKj-pJKit_E.txt"}
    if not all(os.path.exists(os.path.join(tdir, f)) for f in names.values()):
        print("   transcripts not found; skipping (the data files written earlier are kept)")
        return
    test = clean_text(os.path.join(tdir, names["L6"]))
    pre = build_pretraining_text(tdir, names)
    nbytes = len(test)
    print(f"   test text (Lecture 6 transcript, timestamps removed): {nbytes} bytes; text of other lectures: {len(pre)} bytes")

    base = baseline_sizes(test)
    print("   general-purpose compressors on the test text alone (bytes):", base)

    # --- from scratch: the Bayes mixture over the maximal order K = 0..6, and each fixed order
    t0 = time.time()
    m = CharModel(6)
    mix_bits, per_order = m.code_length_bits(test)
    best = per_order.min()
    ident = -(logsumexp(-per_order * LN2) - math.log(7)) / LN2          # = -log2 ( (1/7) sum_K prod_t p_K(x_t|x_<t) )
    assert abs(mix_bits - ident) < 1e-6 * mix_bits
    assert mix_bits <= best + math.log2(7) + 1e-9
    assert abs((mix_bits - best) - math.log2(7)) < 1e-3          # attained here: the best expert's weight tends to 1
    print(f"   from scratch [{time.time()-t0:.0f}s]: mixture {mix_bits/8:.0f} bytes = {mix_bits/nbytes:.3f} bits/char; fixed orders 0..6 (bits/char):",
          np.round(per_order / nbytes, 3), f"; mixture - best fixed order = {mix_bits-best:.2f} bits (bound log2 7 = {math.log2(7):.2f})")
    save("L6_compress_orders.dat", "order bpc", [np.arange(7), per_order / nbytes])

    # --- a real arithmetic coder driven by the same model, and its decoder
    def make_stream(pre_text, seq=None):
        mm = CharModel(6)
        if len(pre_text):
            mm.code_length_bits(pre_text)                                # 'pre-training': the model, incl. the expert weights, has seen it
        mm.vecs = None

        def stream(t, out=None):
            known = out if out is not None else seq
            if t > 0:
                s_prev = known[t - 1]
                mm.update(s_prev, [float(v[s_prev]) for v in mm.vecs])
            mm.vecs = np.stack(mm.expert_vectors())
            pv = mm.weights() @ mm.vecs
            return pv / pv.sum()
        return stream

    def arith_roundtrip(pre_text, label):
        t0 = time.time()
        enc_stream = make_stream(pre_text, seq=list(test))
        bits = ArithmeticCoder.encode(list(test), lambda t: enc_stream(t))
        m2 = CharModel(6)
        if len(pre_text):
            m2.code_length_bits(pre_text)
        ideal, _ = m2.code_length_bits(test)
        t1 = time.time()
        dec_stream = make_stream(pre_text)
        dec = bytes(ArithmeticCoder.decode(bits, nbytes, dec_stream))
        assert dec == bytes(test), "decoding failed"
        print(f"   arithmetic coder, {label}: {len(bits)} bits = {len(bits)/8:.1f} bytes; ideal {ideal:.1f} bits (excess {len(bits)-ideal:.1f} bits); "
              f"decoded text identical [{t1-t0:.0f}s + {time.time()-t1:.0f}s]")
        assert -3 < len(bits) - ideal < 4, (len(bits), ideal)       # arithmetic coding: ideal code length + 0..2 bits
        return len(bits), ideal

    nb0, id0 = arith_roundtrip(b"", "from scratch")

    # --- 'pre-training': the same model after reading other lectures first
    sizes = [0, 4000, 16000, 64000, len(pre)]
    rows = []
    for s in sizes:
        txt = pre[:s]
        mm = CharModel(6)
        if s:
            mm.code_length_bits(txt)
        bits_mix, po = mm.code_length_bits(test)
        if s:
            cs = baseline_sizes(txt)
            cst = baseline_sizes(txt + b" " + test)
            cond = {k: cst[k] - cs[k] for k in cs}                      # conditional proxy: C(S+T) - C(S)
        else:
            cond = base
        rows.append((s, bits_mix / nbytes, cond["xz_raw"] * 8 / nbytes, cond["bz2"] * 8 / nbytes, cond["zlib_raw"] * 8 / nbytes, float(po.min() / nbytes)))
        print(f"   seeing {s:7d} bytes of other lectures first: mixture {rows[-1][1]:.3f} bits/char (best fixed order {rows[-1][5]:.3f}); "
              f"xz {rows[-1][2]:.3f}, bzip2 {rows[-1][3]:.3f}, zlib {rows[-1][4]:.3f}  [conditional proxies C(S+T)-C(S)]")
    arr = np.array(rows)
    save("L6_compress_scaling.dat", "pre mix xz bz2 zlib best", [arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]])
    nb1, id1 = arith_roundtrip(pre, "after seeing all other transcripts")
    if nbytes == 50583 and len(pre) == 268273:        # the numbers typed into the table of Box 'Prediction is compression'
        assert [round(v, 2) for v in arr[:, 1]] == [2.20, 2.12, 2.03, 1.89, 1.74]
        assert [round(v, 2) for v in arr[:, 3]] == [2.24, 2.15, 2.07, 1.93, 1.78]
        assert [round(v, 2) for v in arr[:, 2]] == [2.47, 2.37, 2.29, 2.17, 2.00]
        assert [round(v, 2) for v in arr[:, 4]] == [2.64, 2.57, 2.52, 2.48, 2.47]
        assert (base["zlib_raw"], base["bz2"], base["xz_raw"]) == (16670, 14191, 15607)
        assert round(nb0 / 8) == 13894 and round(nb1 / 8) == 10984
        print("   the numbers typed into the chapter agree with this run")
    else:
        print("   WARNING: transcripts differ from those used for the numbers typed into the chapter")
    with open(os.path.join(OUT, "L6_compress_summary.txt"), "w") as fh:
        fh.write("test_bytes %d\npre_bytes %d\nmix_bits_scratch %.1f\narith_bits_scratch %d\nmix_bits_pre %.1f\narith_bits_pre %d\n" % (nbytes, len(pre), id0, nb0, id1, nb1))
        fh.write("zlib_raw %d\nbz2 %d\nxz_raw %d\n" % (base["zlib_raw"], base["bz2"], base["xz_raw"]))
        fh.write("per_order_bits " + " ".join("%.1f" % v for v in per_order) + "\n")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["arch", "temp", "bt", "arith", "compress"]
    if "arch" in which:
        architecture_checks()
    if "temp" in which:
        temperature_checks()
    if "bt" in which:
        bt_checks()
    if "arith" in which:
        arithmetic()
    if "compress" in which:
        compression_experiment()
    print("\nall requested checks passed")
