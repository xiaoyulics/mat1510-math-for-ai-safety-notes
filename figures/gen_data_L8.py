#!/usr/bin/env python3
"""Lecture 8 -- data files and numerical checks for the notes.

Run:   /opt/anaconda3/bin/python3 -I figures/gen_data_L8.py            (about 1-2 minutes)
       ... --full         one direct simulation of the lecture's own example (N = 200,000 random unit
                          vectors in R^2000: the smallest pairwise angle); several minutes on a GPU
       ... --embeddings   statistics of real token-embedding matrices read from the local Hugging Face
                          cache (read-only; skipped when the files are absent)

Everything quoted in Lecture 8 under "Beyond the lecture" and in the exercises is asserted here.
Closed forms are compared with Monte Carlo; a failed assertion stops the script.

Sections
  A  high-dimensional geometry (the lecture's section 8.6)
       A1 uniform points of the ball; A2 the law of cos(angle) and its moments; A3 sub-Gaussian bound;
       A4 smallest angle among N random vectors (exact tail + Poisson approximation, versus simulation);
       A5 almost orthogonal systems: union bound, Welch bound; A6 effective dimension of anisotropic data
  B  LoRA: identities and parameter counts (Hu et al.; configurations of the models used by Turner et al.)
  C  small models for the discussion of classifiers: base rates, a blind search versus a curriculum
  D  (optional) token embeddings of cached open-weight models

Data files written to figures/data/:  L8_angle_density.dat  L8_angle_std.dat  L8_angle_tail.dat
  L8_minangle.dat  L8_minangle_mc.dat  L8_dimtable.dat  L8_cost.dat  (and L8_embeddings.dat, L8_full.dat)
"""
import argparse
import glob
import json
import math
import os
import struct
import sys
import time

import numpy as np
from scipy import integrate, optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def save(name, header, rows, fmt="%.10g"):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(fmt % v if isinstance(v, (float, np.floating)) else str(v) for v in r) + "\n")
    say("  wrote", os.path.relpath(path, os.path.dirname(HERE)), f"({len(rows)} rows)")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ======================================================================================
#  A. high-dimensional geometry
# ======================================================================================
def log_c(d):
    """log of c_d = Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)), the normalizing constant of the law of cos(angle)."""
    return special.gammaln(d / 2) - 0.5 * math.log(math.pi) - special.gammaln((d - 1) / 2)


def cos_density(d, t):
    """density of T = <U,V> for independent uniform unit vectors in R^d (d >= 3):  c_d (1-t^2)^((d-3)/2)."""
    return np.exp(log_c(d) + 0.5 * (d - 3) * np.log1p(-np.asarray(t) ** 2))


def upper_tail(d, t):
    """P(T >= t) for t >= 0 (exact):  (1/2) I_{1-t^2}((d-1)/2, 1/2)   (regularized incomplete beta)."""
    return 0.5 * special.betainc((d - 1) / 2.0, 0.5, 1.0 - np.asarray(t, dtype=float) ** 2)


def angle_cdf(d, theta_deg):
    """P(angle <= theta) for theta <= 90 degrees."""
    return upper_tail(d, np.cos(np.radians(theta_deg)))


def expected_pairs(d, N, theta_deg):
    """lambda(theta) = C(N,2) P(angle <= theta): expected number of pairs closer than theta."""
    return N * (N - 1) / 2.0 * angle_cdf(d, theta_deg)


def min_angle_quantile(d, N, q=0.5):
    """theta with P(min angle <= theta) = q in the Poisson approximation exp(-lambda(theta)) (degrees)."""
    target = -math.log(1.0 - q)
    f = lambda th: expected_pairs(d, N, th) - target
    lo, hi = 1e-6, 90.0 - 1e-9
    if f(lo) > 0:
        return 0.0
    if f(hi) < 0:
        return 90.0
    return optimize.brentq(f, lo, hi, xtol=1e-10)


def min_angle_mean(d, N):
    """E[min angle] ~ int_0^{90} exp(-lambda(theta)) dtheta (degrees)."""
    f = lambda th: math.exp(-float(expected_pairs(d, N, th)))
    med = min_angle_quantile(d, N, 0.5)
    pts = [max(med - 8, 1e-3), med, min(med + 8, 89.9)]
    v1, _ = integrate.quad(f, 0, pts[0], limit=200)
    v2, _ = integrate.quad(f, pts[0], pts[2], points=[pts[1]], limit=400)
    v3, _ = integrate.quad(f, pts[2], 90.0, limit=200)
    return v1 + v2 + v3


def fixed_vector_quantile(d, N, q=0.5):
    """median angle from one given vector to the nearest of the other N-1 (Poisson approx.)."""
    target = -math.log(1.0 - q)
    f = lambda th: (N - 1) * angle_cdf(d, th) - target
    return optimize.brentq(f, 1e-6, 90 - 1e-9, xtol=1e-10)


def get_gram_max_fn():
    """returns (device_name, function X -> (max off-diagonal cosine)) using torch (MPS/CPU) when available."""
    try:
        import torch

        dev = "mps" if torch.backends.mps.is_available() else "cpu"

        def gram_max(X, block=2000):
            Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
            n = Xt.shape[0]
            best = -2.0
            for i in range(0, n, block):
                G = Xt[i:i + block] @ Xt.T
                idx = torch.arange(i, min(i + block, n), device=dev)
                G[idx - i, idx] = -2.0
                best = max(best, float(G.max().item()))
            return best

        return dev, gram_max
    except Exception:  # numpy fallback
        def gram_max(X, block=1000):
            n = X.shape[0]
            best = -2.0
            for i in range(0, n, block):
                G = X[i:i + block] @ X.T
                G[np.arange(G.shape[0]), np.arange(i, i + G.shape[0])] = -2.0
                best = max(best, float(G.max()))
            return best

        return "cpu-numpy", gram_max


def random_unit(rng, n, d):
    X = rng.standard_normal((n, d), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def part_A(args):
    say("== A. high-dimensional geometry ==")
    rng = np.random.default_rng(8)

    # ---- A1: uniform points of the ball ------------------------------------------------------
    say("A1  P(||X|| > 1-eps) = 1-(1-eps)^d  for X uniform in the unit ball of R^d")
    d, eps = 2000, 1e-3
    p = 1 - (1 - eps) ** d
    say(f"    d={d}, eps={eps}:  1-(1-eps)^d = {p:.5f};   1-exp(-eps d) = {1 - math.exp(-eps * d):.5f}")
    check(abs(p - 0.8648) < 5e-5, "ball shell probability")
    check(p >= 1 - math.exp(-eps * d) - 1e-15, "1-(1-eps)^d >= 1-exp(-eps d)")
    # independent sampler (does not use the radius law): if Y is uniform on S^{d+1} in R^{d+2}, the first d
    # coordinates are uniform in the unit ball B^d (Archimedes / Marsaglia).  Check by Monte Carlo.
    for (dd, ee, n) in [(3, 0.05, 400000), (10, 0.05, 400000), (2000, 1e-3, 40000)]:
        Y = rng.standard_normal((n, dd + 2), dtype=np.float32)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.linalg.norm(Y[:, :dd], axis=1)
        phat = float(np.mean(r > 1 - ee))
        pex = 1 - (1 - ee) ** dd
        se = math.sqrt(pex * (1 - pex) / n)
        say(f"    Monte Carlo (n={n}) d={dd:5d} eps={ee}: simulated {phat:.5f}  exact {pex:.5f}  s.e. {se:.5f}")
        check(abs(phat - pex) < 4.5 * se, f"ball shell MC d={dd}")
    # median radius and the scale of the shell
    med_r = 0.5 ** (1.0 / d)
    say(f"    median radius 2^(-1/d) = {med_r:.6f}  (=1-{(1 - med_r) * 1e4:.3f}e-4);  1-eps with eps = 1-0.1^(1/d): {1 - 0.1 ** (1 / d):.6e}")
    eps90 = 1 - 0.1 ** (1.0 / d)
    check(abs((1 - (1 - eps90) ** d) - 0.9) < 1e-12, "eps for 90 percent")
    # distance to the boundary times d is asymptotically exponential(1)
    for s in (0.5, 1.0, 2.0):
        check(abs((1 - s / d) ** d - math.exp(-s)) < 2e-3, "d*(1-|X|) -> Exp(1)")

    # ---- A2: law of cos(angle) ---------------------------------------------------------------
    say("A2  T = <U,V>: density c_d (1-t^2)^((d-3)/2), E T = 0, E T^2 = 1/d, T^2 ~ Beta(1/2,(d-1)/2)")
    rows_sd = []
    for d in (3, 10, 100, 2000):
        # normalization and variance by quadrature
        norm, _ = integrate.quad(lambda t: float(cos_density(d, t)), -1, 1, epsabs=1e-13, limit=400)
        var, _ = integrate.quad(lambda t: t * t * float(cos_density(d, t)), -1, 1, epsabs=1e-14, limit=400)
        check(abs(norm - 1) < 1e-8, f"density normalization d={d}: {norm}")
        check(abs(var - 1.0 / d) < 1e-9, f"variance d={d}: {var}")
        # simulation: independent pairs
        n = 200000
        U = random_unit(rng, n, d)
        V = random_unit(rng, n, d)
        T = np.sum(U * V, axis=1).astype(np.float64)
        se_mean = T.std() / math.sqrt(n)
        check(abs(T.mean()) < 4.5 * se_mean, f"mean of T d={d}")
        check(abs(T.var() * d - 1) < 0.03, f"var of T d={d}: {T.var() * d}")
        ks = stats.kstest(T, lambda t: 1 - 0.0 * t - np.where(t >= 0, upper_tail(d, np.abs(t)), 1 - upper_tail(d, np.abs(t))))
        check(ks.pvalue > 1e-3, f"KS d={d}: p={ks.pvalue}")
        th = np.degrees(np.arccos(T))
        say(f"    d={d:5d}: quad norm={norm:.10f}  d*Var(T) exact=1, MC={T.var() * d:.4f};  KS p={ks.pvalue:.3f};"
            f"  angle: mean {th.mean():.3f} deg, sd {th.std():.4f} deg  (1/sqrt(d) rad = {math.degrees(1 / math.sqrt(d)):.4f} deg)")
        rows_sd.append((d, math.degrees(1 / math.sqrt(d)), th.std()))
    check(abs(math.degrees(1 / math.sqrt(2000)) - 1.2812) < 1e-3, "1.28 degrees at d=2000")
    # moments  E T^{2k} = prod_{j<k} (2j+1)/(d+2j)
    for d in (7, 30):
        for k in range(1, 6):
            m, _ = integrate.quad(lambda t: t ** (2 * k) * float(cos_density(d, t)), -1, 1, epsabs=1e-14, limit=400)
            ex = np.prod([(2 * j + 1) / (d + 2 * j) for j in range(k)])
            check(abs(m - ex) < 1e-10, f"moment k={k} d={d}")
    say("    moment formula E T^(2k) = prod_{j<k}(2j+1)/(d+2j) verified for d=7,30, k<=5")
    # d = 3: cos(angle) is uniform on [-1,1] (Archimedes)
    check(abs(float(cos_density(3, 0.37)) - 0.5) < 1e-14, "Archimedes")
    # data for the angle-density figures
    ds = (3, 10, 100, 2000)
    th = np.arange(0.0, 180.0001, 0.25)
    rows = []
    for t in th:
        s = math.sin(math.radians(t))
        rows.append((t,) + tuple(float(s ** (d - 2)) if s > 0 else 0.0 for d in ds))
    save("L8_angle_density.dat", ["theta"] + [f"d{d}" for d in ds], rows, fmt="%.8g")
    # standardized: z = sqrt(d) T with density c_d (1 - z^2/d)^((d-3)/2) / sqrt(d)
    zs = np.linspace(-4.0, 4.0, 321)
    rows = []
    for z in zs:
        row = [z]
        for d in ds:
            t = z / math.sqrt(d)
            row.append(float(cos_density(d, t)) / math.sqrt(d) if abs(t) < 1 else 0.0)
        row.append(float(stats.norm.pdf(z)))
        rows.append(tuple(row))
    save("L8_angle_std.dat", ["z"] + [f"d{d}" for d in ds] + ["gauss"], rows, fmt="%.8g")
    # closeness to the Gaussian limit at d = 2000
    dev = max(abs(float(cos_density(2000, z / math.sqrt(2000))) / math.sqrt(2000) - stats.norm.pdf(z)) for z in np.linspace(-4, 4, 81))
    say(f"    standardized density at d=2000 differs from N(0,1) by at most {dev:.2e}")
    check(dev < 2e-3, "Gaussian limit")

    # ---- A3: sub-Gaussian bound ---------------------------------------------------------------
    say("A3  P(T >= t) <= exp(-d t^2 / 2)   (Ball, Lemma 2.2);  proof via E exp(lam T) <= exp(lam^2/(2d))")
    worst = 0.0
    for d in (3, 4, 5, 10, 50, 100, 500, 2000):
        for t in np.linspace(0.001, 0.999, 400):
            lhs = float(upper_tail(d, t))
            rhs = math.exp(-d * t * t / 2)
            check(lhs <= rhs * (1 + 1e-12) + 1e-300, f"sub-Gaussian bound d={d} t={t}")
            if lhs > 0:
                worst = max(worst, lhs / rhs)
    say(f"    verified on a grid (d up to 2000, 400 values of t); the ratio lhs/rhs never exceeded {worst:.3f}")
    for d in (5, 50, 500):
        for lam in (0.5, 2.0, 8.0, 20.0, 40.0):
            if lam * 1.0 > 700:
                continue
            f = lambda t: math.exp(lam * t) * float(cos_density(d, t))
            m, _ = integrate.quad(f, -1, 1, epsabs=1e-12, limit=400)
            check(m <= math.exp(lam * lam / (2 * d)) * (1 + 1e-9), f"mgf d={d} lam={lam}")
    say("    mgf bound E exp(lam T) <= exp(lam^2/(2d)) verified by quadrature on a grid")

    # ---- A4: the smallest angle among N random vectors --------------------------------------------
    say("A4  smallest angle among N random unit vectors in R^d")
    d, N = 2000, 200000
    pairs = N * (N - 1) / 2
    th_med = min_angle_quantile(d, N, 0.5)
    th_1 = optimize.brentq(lambda th: expected_pairs(d, N, th) - 1.0, 1, 89.9)
    th_mean = min_angle_mean(d, N)
    th_fix = fixed_vector_quantile(d, N, 0.5)
    z = stats.norm.isf(1.0 / pairs)
    th_gauss = math.degrees(math.acos(z / math.sqrt(d)))
    say(f"    N={N}, d={d}: pairs={pairs:.4g};  median min angle (Poisson) {th_med:.3f} deg;  lambda=1 at {th_1:.3f} deg;"
        f"  mean {th_mean:.3f} deg;  Gaussian-tail heuristic {th_gauss:.3f} deg")
    say(f"    nearest neighbour of ONE given vector (N-1 competitors): median {th_fix:.3f} deg")
    say(f"    lambda(80 deg) = {expected_pairs(d, N, 80):.3e}  (expected number of pairs below 80 degrees);  P(angle<80)={float(angle_cdf(d, 80)):.3e}")
    check(81.0 < th_med < 82.0 and abs(th_med - 81.66) < 0.02, "81 degrees")
    check(abs(th_fix - 84.24) < 0.02, "fixed vector")
    check(expected_pairs(d, N, 80) < 1e-4, "80 degrees is special")
    # union bound (valid for every N, d): P(min <= th) <= lambda(th)
    th_union = optimize.brentq(lambda th: expected_pairs(d, N, th) - 0.01, 1, 89.9)
    say(f"    union bound: with probability >= 99 percent the smallest angle exceeds {th_union:.3f} deg")
    # subGaussian version of the union bound: lambda <= (N^2/2) exp(-d cos^2/2) = delta  =>  cos = sqrt(2 ln(N^2/(2 delta))/d)
    c_sg = math.sqrt(2 * math.log(N * N / (2 * 0.01)) / d)
    say(f"    sub-Gaussian union bound (delta=0.01): all angles >= {math.degrees(math.acos(c_sg)):.3f} deg")
    check(math.degrees(math.acos(c_sg)) < th_union, "sg bound weaker than exact")
    # dimension at which 200,000 random vectors are all >= 80 degrees apart (median)
    d80 = optimize.brentq(lambda dd: min_angle_quantile(dd, N, 0.5) - 80.0, 300, 4000, xtol=1e-3)
    say(f"    smallest d for which the median min angle among N={N} random vectors reaches 80 degrees: d = {d80:.1f}")

    # --- simulation versus Poisson approximation
    dev_name, gram_max = get_gram_max_fn()
    say(f"    simulation device: {dev_name}")
    combos = [(10, 1000, 200), (30, 1000, 200), (100, 1000, 200), (300, 1000, 200), (1000, 1000, 200),
              (3000, 1000, 200), (10000, 1000, 100),
              (30, 10000, 20), (100, 10000, 20), (300, 10000, 20), (1000, 10000, 20), (3000, 10000, 10),
              (2000, 20000, 10)]
    rows_mc = []
    for (dd, nn, reps) in combos:
        mins = []
        for r in range(reps):
            X = random_unit(rng, nn, dd)
            mins.append(math.degrees(math.acos(min(1.0, gram_max(X)))))
        mins = np.array(mins)
        pm, pmean = min_angle_quantile(dd, nn, 0.5), min_angle_mean(dd, nn)
        se = mins.std(ddof=1) / math.sqrt(reps)
        say(f"    d={dd:6d} N={nn:6d} reps={reps:4d}: simulated min angle {mins.mean():7.3f} +- {se:.3f} deg;"
            f" Poisson mean {pmean:7.3f}, median {pm:7.3f}")
        rows_mc.append((dd, nn, float(mins.mean()), float(se), pmean, pm))
        check(abs(mins.mean() - pmean) < max(4.5 * se, 0.0) + 1.0, f"Poisson approximation off for d={dd}, N={nn}")
    save("L8_minangle_mc.dat", ["d", "N", "sim_mean", "sim_se", "poisson_mean", "poisson_median"], rows_mc, fmt="%.8g")

    # --- the curves of the figure
    ds = np.unique(np.round(np.logspace(math.log10(8), math.log10(20000), 70)).astype(int))
    Ns = (1000, 10000, 100000, 200000)
    rows = []
    for dd in ds:
        rows.append((int(dd),) + tuple(min_angle_quantile(int(dd), n, 0.5) for n in Ns))
    save("L8_minangle.dat", ["d"] + [f"N{n}" for n in Ns], rows, fmt="%.8g")

    # --- tail probability curves P(angle <= theta)
    dsl = (100, 300, 1000, 2000, 5000)
    th = np.arange(30.0, 90.0001, 0.25)
    rows = []
    for t in th:
        row = [t]
        for dd in dsl:
            v = float(angle_cdf(dd, t))
            row.append(v if v > 1e-290 else float("nan"))
        rows.append(tuple(row))
    save("L8_angle_tail.dat", ["theta"] + [f"d{dd}" for dd in dsl], rows, fmt="%.8g")
    say(f"    1/C(N,2) for N={N}: {1 / pairs:.3e}   (the horizontal line of the figure)")

    # --- dimension table
    say("    dimension table")
    rows = []
    for dd in (3, 10, 100, 768, 2000, 2880, 7168, 12000):
        sd_deg = math.degrees(1 / math.sqrt(dd))
        shell = 1 - (1 - 1e-3) ** dd
        m200 = min_angle_quantile(dd, 200000, 0.5)
        p80 = float(angle_cdf(dd, 80.0))
        # sd of angle by quadrature in the exact law (for d=3 it is not 1/sqrt(3) radians exactly)
        tha = lambda t: (math.acos(t) - math.pi / 2) ** 2 * float(cos_density(dd, t))
        v, _ = integrate.quad(tha, -1, 1, limit=400, epsabs=1e-12)
        exact_sd = math.degrees(math.sqrt(v))
        rows.append((dd, exact_sd, shell, p80, m200))
        say(f"      d={dd:6d}: sd(angle) exact {exact_sd:8.4f} deg (1/sqrt(d): {sd_deg:8.4f});  P(|X|>0.999)={shell:.4f};"
            f"  P(angle<80)={p80:.3e};  median min angle (N=2e5) {m200:.2f}")
    save("L8_dimtable.dat", ["d", "sd_deg", "shell", "p80", "minangle_2e5"], rows, fmt="%.8g")

    # ---- A5: almost orthogonal systems ------------------------------------------------------------
    say("A5  how many almost orthogonal directions fit?")
    d = 2000
    th0 = 80.0
    eps0 = math.cos(math.radians(th0))
    n_sg = math.sqrt(2) * math.exp(d * eps0 * eps0 / 4)  # signed version: N^2/2 e^{-d eps^2/2} <= 1
    n_sg_abs = math.exp(d * eps0 * eps0 / 4)  # |cos| <= eps: N^2 e^{-d eps^2/2} <= 1
    # exact: largest N with C(N,2) * P(|T| >= eps) < 1
    p_abs = 2 * float(upper_tail(d, eps0))
    n_exact = 0.5 + math.sqrt(0.25 + 2.0 / p_abs)
    say(f"    d={d}, all pairwise angles in [80,100] degrees (|cos| <= {eps0:.4f}):")
    say(f"      sub-Gaussian union bound guarantees N >= {n_sg_abs:.3g};  exact tail: N >= {n_exact:.3g}  (a random system works with positive probability)")
    check(3.4e6 < n_sg_abs < 3.6e6, "3.5e6")
    check(1e7 < n_exact < 3e7, "exact N")
    welch = lambda N, d: math.sqrt((N - d) / (d * (N - 1)))
    say(f"    Welch bound for N=2e5 unit vectors in R^2000: max |<v_i,v_j>| >= {welch(200000, 2000):.5f}  (angle <= {math.degrees(math.acos(welch(200000, 2000))):.2f} deg for some pair)")
    say(f"    for comparison, random vectors give max |cos| ~ {math.cos(math.radians(min_angle_quantile(2000, 200000, 0.5))):.4f} (signed) ")
    check(abs(welch(200000, 2000) - 0.02225) < 1e-4, "Welch")
    # tiny sanity check of the Welch bound on a random system and on the simplex-like frame (N = d+1)
    Nw, dw = 40, 8
    Xw = random_unit(rng, Nw, dw).astype(np.float64)
    G = np.abs(Xw @ Xw.T - np.eye(Nw))
    check(G.max() >= welch(Nw, dw) - 1e-12, "Welch bound violated?")
    # (N = d+1 simplex: all inner products equal -1/d, attains the Welch bound  sqrt((N-d)/(d(N-1))) = 1/d ... )
    check(abs(welch(dw + 1, dw) - 1.0 / dw) < 1e-12, "simplex attains the Welch bound")

    # ---- A6: effective dimension ---------------------------------------------------------------
    say("A6  effective dimension: E<X,Y>^2 = ||M||_F^2 = 1/d_eff  (M = E[X X^T] for unit vectors, tr M = 1)")
    d, n = 400, 6000
    spec = 1.0 / np.sqrt(np.arange(1, d + 1))  # covariance eigenvalues lambda_k ~ k^(-1/2)
    Z = rng.standard_normal((n, d)) * np.sqrt(spec)
    X = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    M = X.T @ X / n
    deff_emp = 1.0 / np.sum(M * M)
    deff_cov = spec.sum() ** 2 / np.sum(spec ** 2)
    # exact algebraic identity for the empirical measure:  mean over ALL ordered pairs (i,j) of <x_i,x_j>^2 = ||M||_F^2
    gram = X @ X.T
    check(abs(np.mean(gram ** 2) - np.sum(M * M)) < 1e-12, "Frobenius identity")
    i, j = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
    keep = i != j
    cs = np.sum(X[i[keep]] * X[j[keep]], axis=1)
    say(f"    covariance spectrum k^(-1/2), d={d}: d_eff (empirical M) {deff_emp:.1f};  (tr S)^2/tr(S^2) = {deff_cov:.1f};  sd of cos over random pairs {cs.std():.4f}  vs 1/sqrt(d_eff) {1 / math.sqrt(deff_emp):.4f},  1/sqrt(d) {1 / math.sqrt(d):.4f}")
    check(abs(deff_emp / deff_cov - 1) < 0.1, "d_eff vs covariance formula")
    check(abs(cs.std() * math.sqrt(deff_emp) - 1) < 0.05, "sd of cos = 1/sqrt(d_eff)")
    # a shared direction (anisotropy through the mean)
    mu = np.zeros(d)
    mu[0] = 3.0
    X2 = (rng.standard_normal((n, d)) + mu)
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
    xbar = X2.mean(axis=0)
    cs2 = np.sum(X2[i[keep]] * X2[j[keep]], axis=1)
    say(f"    common mean direction (mu = 3 e_1 in R^{d}): mean cos over pairs {cs2.mean():.4f}  vs ||mean unit vector||^2 = {np.dot(xbar, xbar):.4f}")
    check(abs(cs2.mean() - np.dot(xbar, xbar)) < 5e-3, "mean cosine identity")
    return rng


# ======================================================================================
#  B. LoRA
# ======================================================================================
CONFIGS = {
    # name: (hidden, intermediate, layers, heads, kv_heads, vocab, tied, qkv_bias)   (Hugging Face config.json)
    "Qwen2.5-0.5B": (896, 4864, 24, 14, 2, 151936, True, True),
    "Llama-3.2-1B": (2048, 8192, 16, 32, 8, 128256, True, False),
    "Qwen2.5-14B": (5120, 13824, 48, 40, 8, 152064, False, True),
}
HEAD_DIM = {"Qwen2.5-0.5B": 64, "Llama-3.2-1B": 64, "Qwen2.5-14B": 128}


def linear_shapes(name):
    h, inter, L, nh, nkv, V, tied, bias = CONFIGS[name]
    hd = HEAD_DIM[name]
    return {
        "q_proj": (h, nh * hd), "k_proj": (h, nkv * hd), "v_proj": (h, nkv * hd), "o_proj": (nh * hd, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
    }


def total_params(name):
    h, inter, L, nh, nkv, V, tied, bias = CONFIGS[name]
    hd = HEAD_DIM[name]
    sh = linear_shapes(name)
    per_layer = sum(i * o for (i, o) in sh.values()) + 2 * h  # + two RMSNorm weights
    if bias:
        per_layer += nh * hd + 2 * nkv * hd
    emb = V * h
    total = L * per_layer + emb + h + (0 if tied else V * h)
    return total


def lora_params(name, r, modules=None):
    sh = linear_shapes(name)
    L = CONFIGS[name][2]
    mods = modules or list(sh)
    return L * sum(r * (sh[m][0] + sh[m][1]) for m in mods)


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def part_B(rng):
    say("== B. LoRA ==")
    # identities
    d_out, d_in, r, alpha = 64, 48, 4, 8.0
    W = rng.standard_normal((d_out, d_in))
    A = rng.standard_normal((r, d_in)) / math.sqrt(d_in)
    B = np.zeros((d_out, r))
    x = rng.standard_normal((d_in, 5))
    check(np.allclose(W @ x + (alpha / r) * B @ (A @ x), W @ x), "zero init of B: the adapted model starts at the base model")
    B = rng.standard_normal((d_out, r)) * 0.1
    Wm = W + (alpha / r) * B @ A
    check(np.allclose(Wm @ x, W @ x + (alpha / r) * B @ (A @ x)), "merged weights give the same outputs")
    check(np.linalg.matrix_rank(B @ A) <= r, "rank of the update <= r")
    sv = np.linalg.svd((alpha / r) * B @ A, compute_uv=False)
    check(np.sum(sv > 1e-10) == r, "rank exactly r for generic factors")
    say("    verified: W' = W + (alpha/r) B A; B=0 start; merging; rank(BA) <= r")
    # counts for a square matrix
    d, r = 2048, 16
    frac = 2 * d * r / d ** 2
    say(f"    square d x d matrix, d={d}, r={r}: trainable {2 * d * r} vs {d * d} = {100 * frac:.4f} percent;  ratio = 2r/d")
    check(abs(frac - 0.015625) < 1e-12, "1.5625 percent")
    for d, r in ((896, 8), (2048, 16), (5120, 1), (4096, 64)):
        check(2 * d * r == r * (d + d), "2dr")
    # model parameter counts from the configurations (and the safetensors headers of cached files)
    for name in CONFIGS:
        tp = total_params(name)
        say(f"    {name}: total parameters from config = {tp:,}")
    check(total_params("Llama-3.2-1B") == 1235814400, "Llama-3.2-1B has 1,235,814,400 parameters")
    check(total_params("Qwen2.5-0.5B") == 494032768, "Qwen2.5-0.5B has 494,032,768 parameters")
    check(abs(total_params("Qwen2.5-14B") / 1e9 - 14.77) < 0.02, "Qwen2.5-14B has 14.7-14.8 B parameters")
    # cached safetensors header (Qwen2.5-0.5B), read-only
    cands = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/model.safetensors"))
    if cands:
        hdr = read_safetensors_header(cands[0])
        n = sum(int(np.prod(v["shape"])) for k, v in hdr.items() if k != "__metadata__")
        say(f"    Qwen2.5-0.5B safetensors header: {n:,} parameters (count of all stored tensors)")
        check(n == 494032768, "header count equals config count")
    else:
        say("    (no cached Qwen2.5-0.5B file: header check skipped)")
    # adapters
    for name, r in (("Qwen2.5-0.5B", 32), ("Qwen2.5-0.5B", 8), ("Llama-3.2-1B", 32), ("Llama-3.2-1B", 16), ("Llama-3.2-1B", 8)):
        lp = lora_params(name, r)
        say(f"    {name:14s} rank {r:2d} on all 7 linear maps of every layer: {lp:>11,} trainable = {100 * lp / total_params(name):.2f} percent of {total_params(name):,}")
    check(lora_params("Llama-3.2-1B", 16) == 11272192, "Llama-3.2-1B r=16")
    check(lora_params("Llama-3.2-1B", 32) == 22544384, "Llama-3.2-1B r=32")
    check(lora_params("Qwen2.5-0.5B", 32) == 17596416, "Qwen 0.5B r=32")
    # single rank-1 adapter on one MLP down-projection of Qwen2.5-14B (Turner et al., Section 3.5)
    i, o = linear_shapes("Qwen2.5-14B")["down_proj"]
    one = 1 * (i + o)
    say(f"    one rank-1 adapter on a down_proj of Qwen2.5-14B ({i}->{o}): {one:,} trainable parameters = {100 * one / (i * o):.4f} percent of that matrix, "
        f"{one / total_params('Qwen2.5-14B'):.3e} of the model")
    check(one == 18944, "18,944")
    return rng


# ======================================================================================
#  C. small models for the discussion of defences
# ======================================================================================
def part_C(rng):
    say("== C. base rates, blind search versus curriculum ==")
    # base rate
    pi, tpr, fpr = 1e-4, 0.99, 1e-3
    ppv = pi * tpr / (pi * tpr + (1 - pi) * fpr)
    say(f"    prevalence {pi}, TPR {tpr}, FPR {fpr}:  P(malicious | flagged) = {ppv:.4f};  false alarms per true detection = {(1 - ppv) / ppv:.2f}")
    check(abs(ppv - 0.0901) < 5e-4, "ppv")
    fpr2 = 1e-4
    ppv2 = pi * tpr / (pi * tpr + (1 - pi) * fpr2)
    say(f"    with FPR {fpr2}: precision {ppv2:.4f}")
    # FPR needed for precision 1/2
    f_half = pi * tpr * (1 - 0.5) / (0.5 * (1 - pi))
    say(f"    FPR needed for precision 1/2: {f_half:.3e}")
    check(abs(f_half - pi * tpr / (1 - pi)) < 1e-12, "fpr half")
    # blind search versus nested levels: expected number of queries
    rho = 1e-12
    reps = 200000
    g = rng.geometric(1e-3, size=reps)
    say(f"    blind search for a set of density 1e-3: mean queries {g.mean():.1f} (theory 1000)")
    check(abs(g.mean() / 1000 - 1) < 0.02, "geometric mean")
    out = []
    for p in (0.5, 0.2, 0.1, 1 / math.e, 0.05):
        K = math.log(1 / rho) / math.log(1 / p)  # levels, may be non-integer: use ceil in practice
        cost = K / p
        out.append((p, K, cost))
        say(f"    rho=1e-12, per-level pass probability p={p:.3f}: K={K:.1f} levels, expected cost K/p = {cost:.1f}  (blind: {1 / rho:.1e})")
    # optimum over p: minimize ln(1/rho) / (p ln(1/p))  -> p = 1/e, cost e ln(1/rho)
    res = optimize.minimize_scalar(lambda p: math.log(1 / rho) / (p * math.log(1 / p)), bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-10})
    say(f"    optimum p* = {res.x:.4f} (1/e = {1 / math.e:.4f}), cost {res.fun:.2f} = e ln(1/rho) = {math.e * math.log(1 / rho):.2f}")
    check(abs(res.x - 1 / math.e) < 1e-4, "p*=1/e")
    check(abs(res.fun - math.e * math.log(1 / rho)) < 1e-5, "cost at optimum")
    # simulation of the idealized multilevel search: K levels, each needs Geometric(p) trials
    K, p = 12, 0.1
    sim = rng.geometric(p, size=(100000, K)).sum(axis=1)
    say(f"    simulated cost for K={K}, p={p}: mean {sim.mean():.1f} (theory K/p = {K / p:.1f})")
    check(abs(sim.mean() / (K / p) - 1) < 0.01, "multilevel cost")
    # rows for a figure: cost versus log10(1/rho)
    rows = []
    for e in range(1, 31):
        rho = 10.0 ** (-e)
        rows.append((e, 1 / rho, math.e * math.log(1 / rho), math.log(1 / rho) / (0.5 * math.log(2)), math.log(1 / rho) / (0.1 * math.log(10))))
    save("L8_cost.dat", ["log10_inv_rho", "blind", "best", "p_half", "p_tenth"], rows, fmt="%.8g")
    # flag budget: accounts needed
    Fq, k = 330000, 5
    say(f"    flag budget: F={Fq} flags needed, k={k} flags tolerated per account => at least {math.ceil(Fq / k):,} accounts")
    check(math.ceil(Fq / k) == 66000, "accounts")
    # Price/covariance identity for truncation selection: E_sel f - E f = Cov(w, f)  with w = 1/alpha on top fraction
    f = rng.random(10000)
    alpha_sel = 0.2
    thr = np.quantile(f, 1 - alpha_sel)
    w = np.where(f >= thr, 1.0 / np.mean(f >= thr), 0.0)
    gain = np.sum(w * f) / len(f) - f.mean()
    cov = np.mean(w * f) - np.mean(w) * np.mean(f)
    check(abs(np.mean(w) - 1) < 1e-12 and abs(gain - cov) < 1e-12, "Price identity")
    say(f"    Price identity: gain of the mean fitness under selection = Cov(w,f): {gain:.5f} = {cov:.5f}; constant fitness gives 0")


# ======================================================================================
#  D. optional: embeddings of cached models (read-only)
# ======================================================================================
EMB_FILES = [
    # label, glob pattern (relative to the Hugging Face hub cache), tensor names to try, tokenizer.json pattern
    ("GPT-2 (wte, tied)", "models--gpt2/snapshots/*/model.safetensors", ["wte.weight", "transformer.wte.weight"], "models--gpt2/snapshots/*/tokenizer.json"),
    ("Pythia-410m (embed_in)", "models--EleutherAI--pythia-410m/snapshots/*/model.safetensors", ["gpt_neox.embed_in.weight"], "models--EleutherAI--pythia-410m/snapshots/*/tokenizer.json"),
    ("Pythia-410m (embed_out)", "models--EleutherAI--pythia-410m/snapshots/*/model.safetensors", ["embed_out.weight"], "models--EleutherAI--pythia-410m/snapshots/*/tokenizer.json"),
    ("Qwen2.5-Coder-1.5B (tied)", "models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/model.safetensors", ["model.embed_tokens.weight"], "models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/*/tokenizer.json"),
    ("Qwen2.5-Coder-3B (tied)", "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/*/model-0000*-of-00002.safetensors", ["model.embed_tokens.weight"], "models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/*/tokenizer.json"),
]


def hub_glob(pattern):
    return sorted(glob.glob(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), pattern)))


def load_tensor(pattern, names):
    from safetensors import safe_open

    for path in hub_glob(pattern):
        with safe_open(path, framework="pt") as f:  # bfloat16 is not understood by numpy
            for nm in names:
                if nm in f.keys():
                    return f.get_tensor(nm).float().numpy()
    return None


def tokenizer_size(pattern):
    """number of token ids of the tokenizer (rows beyond it are padding of the embedding matrix)"""
    paths = hub_glob(pattern)
    if not paths:
        return None
    tk = json.load(open(paths[0]))
    ids = list(tk["model"]["vocab"].values()) if isinstance(tk["model"]["vocab"], dict) else []
    ids += [t["id"] for t in tk.get("added_tokens", [])]
    return max(ids) + 1


def gram_pass(X, thr_cos, dup_cos=None, block=2048):
    """one pass over the Gram matrix of unit rows X (torch tensor).
    returns (number of unordered pairs with cos > thr_cos, per-row max cosine to other rows, per-row count of cos > dup_cos)"""
    import torch

    n = X.shape[0]
    count = 0
    nn_max = torch.empty(n, device=X.device)
    dup = torch.zeros(n, dtype=torch.int32, device=X.device)
    for i in range(0, n, block):
        G = X[i:i + block] @ X.T
        idx = torch.arange(i, min(i + block, n), device=X.device)
        G[idx - i, idx] = -2.0
        count += int((G > thr_cos).sum().item())
        nn_max[i:i + block] = G.max(dim=1).values
        if dup_cos is not None:
            dup[i:i + block] = (G > dup_cos).sum(dim=1).to(torch.int32)
    return count // 2, nn_max, dup


def embedding_stats(E, dev):
    """statistics of the directions of the rows of E (V x d): see Section 'Real embeddings are not random'."""
    import torch

    V, d = E.shape
    out = {}
    cth = math.cos(math.radians(80.0))
    for variant in ("raw", "centered"):
        A = E - E.mean(axis=0, keepdims=True) if variant == "centered" else E
        nr = np.linalg.norm(A, axis=1)
        X = torch.as_tensor(A / nr[:, None], device=dev)
        n = X.shape[0]
        M = (X.T @ X) / n
        fro2 = float(torch.sum(M * M).item())
        xbar = X.mean(dim=0)
        count, nn_max, _ = gram_pass(X, cth)
        pairs = n * (n - 1) / 2.0
        nn_ang = torch.rad2deg(torch.acos(nn_max.clamp(-1, 1))).cpu().numpy()
        out[variant] = dict(n=n, d=d, mean_cos=float(torch.dot(xbar, xbar).item()), rms_cos=math.sqrt(fro2), d_eff=1 / fro2,
                            min_angle=float(nn_ang.min()), frac80=count / pairs, med_nn=float(np.median(nn_ang)), tok80=float(np.mean(nn_ang < 80)))
        del X, M, nn_max
    return out


def part_D(args):
    say("== D. token embeddings of cached open-weight models (read-only) ==")
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for (label, pattern, names, tokpat) in EMB_FILES:
        E = load_tensor(pattern, names)
        if E is None:
            say(f"    {label}: files not found in the Hugging Face cache; skipped")
            continue
        E = np.asarray(E, dtype=np.float32)
        V0 = E.shape[0]
        ntok = tokenizer_size(tokpat)
        if ntok is not None and ntok < V0:
            E = E[:ntok]
        d = E.shape[1]
        # distinct tokens: drop rows that coincide (cos > 0.9999) with another row (untrained tokens share their initial value)
        X = torch.as_tensor(E / np.linalg.norm(E, axis=1, keepdims=True), device=dev)
        _, _, dup = gram_pass(X, 2.0, dup_cos=0.9999)
        keep = (dup == 0).cpu().numpy()
        n_dropped = int((~keep).sum())
        del X
        E = E[keep]
        nq = np.quantile(np.linalg.norm(E, axis=1), [0.05, 0.5, 0.95])
        say(f"    {label}: matrix {V0} x {d}; tokenizer ids {ntok}; rows used {E.shape[0]} (dropped {n_dropped} rows that coincide with another row);"
            f" row norms: 5% {nq[0]:.3f}, median {nq[1]:.3f}, 95% {nq[2]:.3f}")
        st = embedding_stats(E, dev)
        for variant in ("raw", "centered"):
            s = st[variant]
            say(f"        {variant:8s}: mean cos {s['mean_cos']:.4f}; rms cos {s['rms_cos']:.4f} (1/sqrt(d) = {1 / math.sqrt(d):.4f}); d_eff = {s['d_eff']:.1f};"
                f" pairs below 80 deg: {s['frac80']:.3e} (isotropic random {float(angle_cdf(d, 80.0)):.3e}); smallest angle {s['min_angle']:.2f} deg;"
                f" median nearest-neighbour angle {s['med_nn']:.1f} deg; tokens with a neighbour below 80 deg: {100 * s['tok80']:.1f} percent")
            rows.append((label.replace(" ", "_"), variant, s["n"], d, s["mean_cos"], s["rms_cos"], 1 / math.sqrt(d), s["d_eff"],
                         s["min_angle"], s["frac80"], float(angle_cdf(d, 80.0)), s["med_nn"], s["tok80"], n_dropped,
                         float(nq[0]), float(nq[1]), float(nq[2])))
    if rows:
        save("L8_embeddings.dat",
             ["model", "variant", "V", "d", "mean_cos", "rms_cos", "inv_sqrt_d", "d_eff", "min_angle", "frac_below80", "random_frac_below80",
              "median_nn_angle", "frac_tokens_nn_below80", "dropped", "norm_p5", "norm_median", "norm_p95"],
             rows, fmt="%.6g")


def part_full(args):
    say("== optional: direct simulation of the lecture's example ==")
    dev_name, gram_max = get_gram_max_fn()
    rng = np.random.default_rng(2026)
    N, d = 200000, 2000
    rows = []
    for rep in range(args.full_reps):
        t0 = time.time()
        X = random_unit(rng, N, d)
        c = gram_max(X, block=2000)
        ang = math.degrees(math.acos(c))
        say(f"    replicate {rep}: N={N}, d={d}: largest cosine {c:.5f}, smallest angle {ang:.3f} deg  ({time.time() - t0:.0f} s on {dev_name})")
        rows.append((rep, N, d, c, ang))
        check(abs(ang - 81.66) < 0.6, "full simulation vs prediction")
        del X
    save("L8_full.dat", ["rep", "N", "d", "max_cos", "min_angle"], rows, fmt="%.8g")


def exercise_checks():
    """numbers quoted in the hints to the exercises of Lecture 8"""
    say("== exercise checks ==")
    d = 2000
    # 1
    e50 = 1 - 0.5 ** (1 / d)
    e90 = 1 - 0.1 ** (1 / d)
    check(abs(e50 - 3.465e-4) < 1e-7 and abs(e90 - 1.1506e-3) < 1e-6, "ex1 eps")
    check(abs(0.5 ** (1 / d) - 0.999653) < 1e-6, "ex1 median radius")
    say(f"  1: eps50={e50:.4e}, eps90={e90:.4e}, median radius={0.5 ** (1 / d):.6f}")
    # 2
    sds = [math.degrees(1 / math.sqrt(x)) for x in (100, 2000, 12000)]
    check(abs(sds[0] - 5.730) < 1e-3 and abs(sds[1] - 1.2812) < 1e-4 and abs(sds[2] - 0.5230) < 1e-4, "ex2 sd")
    dmin = (180 / math.pi) ** 2
    check(math.ceil(dmin) == 3283 and abs(dmin - 3282.8) < 0.05, "ex2 d")
    say(f"  2: sd = {sds[0]:.3f}, {sds[1]:.4f}, {sds[2]:.4f} degrees; sd < 1 degree iff d > {dmin:.1f}")
    # 3
    N = 200000
    c_sg = math.sqrt(2 * math.log(N * N / (2 * 0.01)) / d)
    a_sg = math.degrees(math.acos(c_sg))
    a_ex = optimize.brentq(lambda th: expected_pairs(d, N, th) - 0.01, 1, 89.9)
    check(abs(a_sg - 80.31) < 0.01 and abs(a_ex - 80.88) < 0.01, "ex3")
    say(f"  3: sub-Gaussian union bound {a_sg:.3f} deg; exact union bound {a_ex:.3f} deg")
    # 4
    m_fixed = fixed_vector_quantile(d, N, 0.5)
    m_all = min_angle_quantile(d, N, 0.5)
    check(abs(m_fixed - 84.24) < 0.01 and abs(m_all - 81.66) < 0.01, "ex4")
    say(f"  4: fixed vector {m_fixed:.2f} deg; all pairs {m_all:.2f} deg")
    # 5
    eps0 = math.cos(math.radians(80))
    check(abs(math.exp(d * eps0 ** 2 / 4) - 3.53e6) < 0.01e6, "ex5 packing")
    check(abs(math.sqrt((N - d) / (d * (N - 1))) - 0.02225) < 1e-5, "ex5 welch")
    say(f"  5: e^(d eps^2/4) = {math.exp(d * eps0 ** 2 / 4):.3e}; Welch bound {math.sqrt((N - d) / (d * (N - 1))):.5f}")
    # 6
    deff = 1 / 0.0388 ** 2
    check(abs(deff - 664.2) < 0.5, "ex6")
    say(f"  6: d_eff = {deff:.1f}")
    # 7
    lp = lora_params("Llama-3.2-1B", 16)
    check(lp == 11272192 and abs(100 * lp / 1235814400 - 0.9121) < 1e-3, "ex7")
    say(f"  7: {lp:,} = {100 * lp / 1235814400:.4f} percent")
    # 8, 9
    pi, tpr, fpr = 1e-4, 0.99, 1e-3
    ppv = pi * tpr / (pi * tpr + (1 - pi) * fpr)
    check(abs(ppv - 0.0901) < 5e-4, "ex8")
    check(abs(pi * tpr / (1 - pi) - 9.901e-5) < 1e-7, "ex8 fpr")
    check(abs(math.e * math.log(1e12) - 75.11) < 0.01, "ex9")
    check(math.ceil(330000 / 5) == 66000, "ex9 accounts")
    say(f"  8: precision {ppv:.4f}; fpr for 1/2: {pi * tpr / (1 - pi):.3e};  9: e ln(1/rho) = {math.e * math.log(1e12):.2f}; accounts {math.ceil(330000 / 5)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--full-reps", type=int, default=2)
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--skip-base", action="store_true", help="skip sections A-C (use with --full/--embeddings)")
    args = ap.parse_args()
    t0 = time.time()
    if not args.skip_base:
        rng = part_A(args)
        part_B(rng)
        part_C(rng)
        exercise_checks()
    if args.embeddings:
        part_D(args)
    if args.full:
        part_full(args)
    say(f"all checks passed ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
