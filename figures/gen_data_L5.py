#!/usr/bin/env python3
"""Checks and figure data for the Lecture 5 notes (Bayes, SGD, grokking).

Run from the project root:   python3 figures/gen_data_L5.py            (about 8 minutes: the last part retrains 4 networks)
                             python3 figures/gen_data_L5.py --quick    (skips the PyTorch part, about 20 seconds)
Needs numpy and scipy; the training part also needs PyTorch.  It complements figures/gen_data.py (group 5: exact Gaussian
regret vs Monte Carlo, Langevin histogram, Fourier partial sums) and figures/grokking.py (the grokking experiment), which
it does not modify.  Every number quoted in Lecture 5 that is not produced by those two scripts is asserted here.

Output (figures/data/):  L5_precision.dat, L5_prior_cost.dat, L5_info.dat, and (without --quick) L5_grok_profile.dat, L5_profile_spectrum.dat.

Contents
  A. Gaussian example: closed form of the Bayes mixture (quadrature), exact regret for a prior N(0, s^2 I) (Gauss-Hermite),
     regret as the sum of the one-step KL errors, comparison with Clarke-Barron, effect of the prior scale.
  B. Solomonoff-style discretization: the precision/description trade-off, an explicit prefix code (Elias delta) and the
     resulting oracle bound compared with the Gaussian prior (figure data).
  C. MAP: closed form and sequential plug-in regret.
  D. Langevin: Fokker-Planck stationarity (symbolic), stationary variance of Euler-Maruyama / SGD / SGLD on a quadratic.
  E. Accuracy versus loss (numbers of the lecture).
  F. Fourier identity and an explicit width-4(p-1) square-activation network for (a+b) mod p.
  G. Neuron statistics from grokking_neurons.dat; (slow) the offset profile of the held-out logits along training (what the
     generalizing part of the network computes, how large its signal-to-noise ratio is, how many frequencies it uses).
"""
import math
import os
import sys

import numpy as np
from scipy import integrate, optimize, special

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
QUICK = "--quick" in sys.argv
GROK_ONLY = "--grok-only" in sys.argv       # run only part G2 (the PyTorch part) and merge its numbers into L5_info.dat
INFO = {}                      # numbers quoted in the notes (written to L5_info.dat)


def save(name, header, cols):
    arr = np.column_stack(cols)
    np.savetxt(os.path.join(OUT, name), arr, header=header, comments="", fmt="%.8e")
    print(f"wrote figures/data/{name}  shape={arr.shape}")


if (GROK_ONLY or QUICK) and os.path.exists(os.path.join(OUT, "L5_info.dat")):      # keep the numbers of the part that is not recomputed
    INFO.update(dict(l.split(None, 1) for l in open(os.path.join(OUT, "L5_info.dat")).read().splitlines() if l.strip()))


def explicit_network(p):
    """Hidden units (u_i, v_i) in R^p x R^p and output weights A (p x H); h_i(m,n) = (u_i[m] + v_i[n])^2; f = A h.
    For each frequency k = 1..(p-1)/2 and a = cos(w k m), b = cos(w k n) etc. use the polarization identity xy = [(x+y)^2 - (x-y)^2]/4:
       cos(w k (m+n)) = cm cn - sm sn,   sin(w k (m+n)) = sm cn + cm sn   (8 hidden units per frequency)."""
    w = 2 * math.pi / p
    m = np.arange(p)
    U, V, A = [], [], []
    for k in range(1, (p - 1) // 2 + 1):
        c, s = np.cos(w * k * m), np.sin(w * k * m)
        cj, sj = np.cos(w * k * m), np.sin(w * k * m)                              # cos/sin(w k j) as functions of the output index j
        # (x, y, sign, which output pattern): product x(m) y(n) = [(x+y)^2 - (x-y)^2]/4
        # y_j = (2/p) sum_k [ cos(w k (m+n)) cos(w k j) + sin(w k (m+n)) sin(w k j) ]
        #     = (2/p) sum_k [ (cm cn - sm sn) cj + (sm cn + cm sn) sj ]
        terms = [(c, c, +1.0, cj), (s, s, -1.0, cj), (s, c, +1.0, sj), (c, s, +1.0, sj)]
        for x, yv, sign, out in terms:
            for sgn_in, sgn_out in ((+1.0, +1.0), (-1.0, -1.0)):                  # (x + y)^2 with + , (x - y)^2 with -
                U.append(x)
                V.append(sgn_in * yv)
                A.append((2.0 / p) * sign * sgn_out * 0.25 * out)
    return np.array(U), np.array(V), np.array(A).T                                 # (H x p), (H x p), (p x H)


if not GROK_ONLY:
    # ----------------------------------------------------------------------------------------------
    # A. Gaussian example.  q = N(theta*, I_d); model N(theta, I_d); prior N(0, s2 I_d).   All logarithms natural (nats).
    # ----------------------------------------------------------------------------------------------
    def log_pB_closed(X, s2=1.0):
        """log p_B(D_n) for data X (n x d) and prior N(0, s2 I):  -nd/2 log 2pi - d/2 log(1+n s2) - 1/2 sum|x|^2 + n^2|xbar|^2/(2a), a = n + 1/s2."""
        n, d = X.shape
        a = n + 1.0 / s2
        xbar = X.mean(axis=0)
        return (-0.5 * n * d * math.log(2 * math.pi) - 0.5 * d * math.log(1 + n * s2)
                - 0.5 * np.sum(X ** 2) + n ** 2 * np.sum(xbar ** 2) / (2 * a))


    def log_pB_quadrature(X, s2=1.0):
        """log of  int pi(theta) prod_i N(x_i; theta, I) dtheta  by direct numerical integration (d = 1 or 2)."""
        n, d = X.shape
        if d == 1:
            f = lambda t: math.exp(-0.5 * t * t / s2) / math.sqrt(2 * math.pi * s2) * math.exp(
                -0.5 * np.sum((X[:, 0] - t) ** 2)) / (2 * math.pi) ** (n / 2)
            val, err = integrate.quad(f, -12, 12, epsabs=0, epsrel=1e-12, limit=400)
        else:
            f = lambda t2, t1: math.exp(-0.5 * (t1 * t1 + t2 * t2) / s2) / (2 * math.pi * s2) * math.exp(
                -0.5 * np.sum((X[:, 0] - t1) ** 2 + (X[:, 1] - t2) ** 2)) / (2 * math.pi) ** (n * 2 / 2)
            val, err = integrate.dblquad(f, -12, 12, -12, 12, epsabs=0, epsrel=1e-10)
        return math.log(val)


    rng = np.random.default_rng(5)
    for d, n, s2 in [(1, 3, 1.0), (1, 7, 4.0), (2, 4, 1.0), (2, 6, 0.5)]:
        X = rng.standard_normal((n, d)) + 0.7
        a, b = log_pB_closed(X, s2), log_pB_quadrature(X, s2)
        assert abs(a - b) < 1e-7, (d, n, s2, a, b)
    print("A1  closed form of p_B(D_n) agrees with numerical integration (d = 1, 2; prior variances 1, 4, 0.5)")


    def regret_exact(n, d, th2, s2=1.0):
        """R_n = E_q log q(D_n)/p_B(D_n) for prior N(0, s2 I):  d/2 ln(1+n s2) + n (|theta*|^2 - s2 d) / (2 (1 + n s2))."""
        return 0.5 * d * np.log1p(n * s2) + n * (th2 - s2 * d) / (2 * (1 + n * s2))


    def regret_gauss_hermite(n, d, th2, s2=1.0, deg=80):
        """Independent evaluation: log q/p_B depends on the data only through xbar (the |x_i|^2 terms cancel against q's);
        E over xbar ~ N(theta*, I/n), done coordinatewise by Gauss-Hermite for d = 1 and exactly for general d via moments."""
        a = n + 1.0 / s2
        # log q/p_B = d/2 ln(1+n s2) + n theta*.xbar - n/2|theta*|^2 - n^2 |xbar|^2/(2a)
        # with xbar = theta* + z/sqrt(n):  E|xbar|^2 = |theta*|^2 + d/n ,  E theta*.xbar = |theta*|^2   (moments, exact)
        # -> verify the d = 1 case with a genuine quadrature of the function:
        if d == 1:
            th = math.sqrt(th2)
            x, w = np.polynomial.hermite_e.hermegauss(deg)           # weight exp(-x^2/2)
            xbar = th + x / math.sqrt(n)
            vals = 0.5 * math.log(1 + n * s2) + n * th * xbar - 0.5 * n * th ** 2 - n ** 2 * xbar ** 2 / (2 * a)
            return float(np.sum(w * vals) / math.sqrt(2 * math.pi))
        return float(0.5 * d * math.log(1 + n * s2) + n * th2 - 0.5 * n * th2 - n ** 2 * (th2 + d / n) / (2 * a))


    for n, th2, s2 in [(1, 0.0, 1.0), (10, 4.0, 1.0), (100, 9.0, 1.0), (1000, 2.0, 25.0), (50, 0.3, 0.1)]:
        assert abs(regret_exact(n, 1, th2, s2) - regret_gauss_hermite(n, 1, th2, s2)) < 1e-9
        assert abs(regret_exact(n, 3, th2, s2) - regret_gauss_hermite(n, 3, th2, s2)) < 1e-9
    print("A2  exact regret R_n = d/2 ln(1+n s2) + n(|th*|^2 - s2 d)/(2(1+n s2)) confirmed by Gauss-Hermite quadrature")

    # Monte Carlo of the whole log-ratio (a third, fully independent route; 6 standard errors)
    for d, n, th2, s2 in [(2, 15, 3.0, 1.0), (1, 40, 6.25, 9.0), (3, 8, 0.5, 0.25)]:
        th = np.zeros(d); th[0] = math.sqrt(th2)
        M = 300000
        X = th + rng.standard_normal((M, n, d))
        a = n + 1.0 / s2
        logq = -0.5 * np.sum((X - th) ** 2, axis=(1, 2))
        xbar = X.mean(axis=1)
        logpB = -0.5 * d * math.log(1 + n * s2) - 0.5 * np.sum(X ** 2, axis=(1, 2)) + n ** 2 * np.sum(xbar ** 2, axis=1) / (2 * a)
        mc = logq - logpB
        assert abs(mc.mean() - regret_exact(n, d, th2, s2)) < 6 * mc.std() / math.sqrt(M)
    print("A3  Monte Carlo of E log q/p_B agrees (prior scales 1, 9, 0.25)")

    # regret = sum of one-step KL errors of the predictive N(n xbar/(n+1), (1 + 1/(n+1)) I)   (chain rule, Lecture 3)
    d, th2, N = 3, 5.0, 60
    steps = []
    for n in range(N):
        s2p = (n + 2.0) / (n + 1.0)                                     # predictive variance 1 + 1/(n+1)
        e_sq = (th2 + n * d) / (n + 1.0) ** 2                           # E|theta* - n xbar/(n+1)|^2
        steps.append(0.5 * (d * (1 / s2p - 1 + math.log(s2p)) + e_sq / s2p))
    assert abs(sum(steps) - regret_exact(N, d, th2)) < 1e-10
    for N2 in (1, 2, 5):
        assert abs(sum(steps[:N2]) - regret_exact(N2, d, th2)) < 1e-10
    for n_ in (1, 2, 10, 1000):
        dd_, th2_ = 3, 5.0
        lhs = regret_exact(n_ + 1, dd_, th2_) - regret_exact(n_, dd_, th2_)
        rhs = 0.5 * dd_ * math.log1p(1.0 / (n_ + 1)) + (th2_ - dd_) / (2 * (n_ + 1) * (n_ + 2))
        assert abs(lhs - rhs) < 1e-12
        if n_ >= 10:
            assert abs(lhs * 2 * n_ / dd_ - 1) < 10.0 / n_          # R_{n+1} - R_n ~ d/(2n)
    print("A4  R_N = sum_{n<N} E KL(q || predictive_n) checked for N = 1, 2, 5, 60")

    # comparison with Clarke-Barron:  d/2 ln(n/2 pi e) + ln( sqrt(det I)/pi(theta*) ),  I = identity, pi = N(0, I)
    for d, th2 in [(1, 0.0), (1, 4.0), (3, 7.0)]:
        for n in (10 ** 3, 10 ** 5):
            cb = 0.5 * d * math.log(n / (2 * math.pi * math.e)) + 0.5 * d * math.log(2 * math.pi) + 0.5 * th2
            gap = regret_exact(n, d, th2) - cb
            assert abs(gap + (th2 - 2 * d) / (2 * n)) < 3.0 * (d + th2 + 1) / n ** 2 * 10, (d, th2, n, gap)
    print("A5  R_n - Clarke-Barron = -(|th*|^2 - 2d)/(2n) + O(n^-2) (so they agree up to O(1/n))")

    # prior scale: R_n - d/2 ln n -> d ln s + |th*|^2/(2 s^2) - d/2 ; minimized at s^2 = |th*|^2/d
    n, d, th2 = 10 ** 7, 2, 12.0
    for s in (0.5, 1.0, 3.0, 7.0):
        pred = d * math.log(s) + th2 / (2 * s * s) - 0.5 * d
        assert abs(regret_exact(n, d, th2, s * s) - 0.5 * d * math.log(n) - pred) < 1e-5
    s_opt = math.sqrt(th2 / d)
    res = optimize.minimize_scalar(lambda s: d * math.log(s) + th2 / (2 * s * s), bounds=(0.1, 20), method="bounded", options={"xatol": 1e-10})
    assert abs(res.x - s_opt) < 1e-5
    print("A6  large-n charge of the prior N(0,s^2 I):  d ln s + |th*|^2/(2 s^2) - d/2, minimal at s^2 = |th*|^2/d  (value d/2 ln(|th*|^2/d))")

    # numbers for the exercises
    INFO["ex2_R_n10_d1_th2"] = float(regret_exact(10, 1, 4.0))
    INFO["ex2_R_n100_d3_th5"] = float(regret_exact(100, 3, 25.0))
    print("   exercise values: R_10(d=1,|th*|=2) = %.4f ; R_100(d=3,|th*|=5) = %.4f" % (INFO["ex2_R_n10_d1_th2"], INFO["ex2_R_n100_d3_th5"]))
    # exercise 5: large-n charge d ln s + |th*|^2/(2 s^2) - d/2 for d = 1, |th*| = 10 and s = 1, 3, 10 (best s = |th*|/sqrt(d) = 10)
    for s_, val in ((1.0, 49.5), (3.0, 6.1542), (10.0, 2.3026)):
        ch = math.log(s_) + 100.0 / (2 * s_ * s_) - 0.5
        assert abs(ch - val) < 1e-4, (s_, ch)
        n_big = 10 ** 12
        assert abs(regret_exact(n_big, 1, 100.0, s_ * s_) - 0.5 * math.log(n_big) - ch) < 1e-6
    print("   exercise 5 values (d = 1, |th*| = 10): charge 49.5 (s=1), 6.154 (s=3), 2.303 (s=10 = best)")
    # exercise 4: f(delta) minimum for n=100, d=1, c=3
    assert abs((0.5 + math.log(3) + 0.5 * math.log(100)) - 3.9012) < 1e-4

    # ----------------------------------------------------------------------------------------------
    # B. Discretization (Solomonoff style)
    # ----------------------------------------------------------------------------------------------
    # B1: f(delta) = (n d/2) delta^2 + d ln(c/delta) is minimal at delta = n^-1/2 with value d/2 + d ln c + d/2 ln n.
    n, d, c = 100, 1, 3.0
    dl = np.logspace(-2, 0, 201)
    dist = 0.5 * n * d * dl ** 2
    desc = d * np.log(c / dl)
    tot = dist + desc
    res = optimize.minimize_scalar(lambda x: 0.5 * n * d * x * x + d * math.log(c / x), bounds=(1e-3, 1.0), method="bounded", options={"xatol": 1e-12})
    assert abs(res.x - n ** -0.5) < 1e-6 and abs(res.fun - (0.5 * d + d * math.log(c) + 0.5 * d * math.log(n))) < 1e-9
    INFO["B_fmin_n100_d1_c3"] = float(res.fun)
    save("L5_precision.dat", "delta distortion description total", [dl, dist, desc, tot])
    print("B1  precision trade-off: minimum at delta = n^-1/2 = %.3f with value %.4f" % (n ** -0.5, res.fun))


    # B2: an explicit prefix-free code: Elias delta for integers >= 1.  Verified to be prefix free (Kraft sum < 1) and its length formula.
    def elias_gamma(N):
        b = bin(N)[2:]
        return "0" * (len(b) - 1) + b


    def elias_delta(N):
        b = bin(N)[2:]
        return elias_gamma(len(b)) + b[1:]


    def delta_len(N):
        L = N.bit_length() - 1
        return L + 2 * ((L + 1).bit_length() - 1) + 1


    codes = [elias_delta(N) for N in range(1, 20001)]
    assert all(len(codes[N - 1]) == delta_len(N) for N in range(1, 20001))
    assert sum(2.0 ** -len(w) for w in codes) < 1.0
    srt = sorted(codes)
    assert all(not srt[i + 1].startswith(srt[i]) for i in range(len(srt) - 1))     # prefix free (sorted neighbours suffice)
    print("B2  Elias delta code verified prefix free for 1..20000; length = L + 2 floor(log2(L+1)) + 1, L = floor(log2 N)")


    def zz(m):
        """bijection Z -> {1,2,3,...}:  0, -1, 1, -2, 2, ...  ->  1, 2, 3, 4, 5, ... (the sign is absorbed into the integer, so the code stays prefix free)"""
        return 2 * m + 1 if m >= 0 else -2 * m


    def universal_bound_1d(n, th, kmax=59):
        """min over 0 <= k <= kmax of  (n/2) e_k^2 + ln2 * len(code)   for the hypotheses N(m 2^-k, 1), m in Z:
        theta~ = sign * floor(|th| 2^k) 2^-k (truncation, as in the lecture), codeword = delta(k+1) . delta(zz(m)), a concatenation of two
        Elias delta codewords, hence prefix free; the prior weights 2^-len sum to at most 1 (Kraft; exactly 1 for the Elias delta code), so  p_B >= 2^-len(h) p_h  and
            R_n  <=  E_q log q/p_h + len(h) ln 2 = (n/2)|theta - theta~|^2 + len(h) ln 2   for every hypothesis h in the class."""
        best = (float("inf"), None)
        sg = 1 if th >= 0 else -1
        for k in range(0, kmax + 1):
            m = sg * math.floor(abs(th) * 2 ** k)
            e = th - m * 2.0 ** -k
            bits = delta_len(k + 1) + delta_len(zz(m))
            val = 0.5 * n * e * e + math.log(2) * bits
            if val < best[0]:
                best = (val, k)
        return best


    # oracle bound check against the regret of the mixture over (a finite part of) this countable class, d = 1, by Monte Carlo.
    # Restricting the mixture to k <= KM, |m 2^-k| <= TM lowers p_B, i.e. raises the regret, and the bound for the restricted class (kmax = KM) must still hold.
    KM, TM = 7, 10.0


    def mixture_regret_mc(n, th, M=3000, seed=0):
        r = np.random.default_rng(seed)
        X = th + r.standard_normal((M, n))
        S1 = X.sum(axis=1)
        S2 = (X ** 2).sum(axis=1)
        logs = []
        for k in range(0, KM + 1):
            ms = np.arange(-int(TM * 2 ** k), int(TM * 2 ** k) + 1)
            t = ms * 2.0 ** -k
            bits = delta_len(k + 1) + np.array([delta_len(zz(int(m))) for m in ms])
            logw = -bits * math.log(2)
            lp = -0.5 * n * math.log(2 * math.pi) - 0.5 * (S2[:, None] - 2 * t[None, :] * S1[:, None] + n * t[None, :] ** 2)
            logs.append(special.logsumexp(lp + logw[None, :], axis=1))
        logpB = special.logsumexp(np.stack(logs, axis=1), axis=1)
        logq = -0.5 * n * math.log(2 * math.pi) - 0.5 * (S2 - 2 * th * S1 + n * th ** 2)
        v = logq - logpB
        return float(v.mean()), float(v.std() / math.sqrt(M))


    for n_, th_ in [(10, 0.37), (50, 2.6), (200, 5.1)]:
        mc, se = mixture_regret_mc(n_, th_)
        ub, k_ = universal_bound_1d(n_, th_, kmax=KM)
        assert mc - 4 * se <= ub + 1e-9, (n_, th_, mc, ub)
        print("   oracle bound check: n=%d theta=%.2f  regret of the Elias-delta mixture %.3f +- %.3f  <=  bound %.3f (k=%d)" % (n_, th_, mc, se, ub, k_))

    # B3: figure data: the price of |theta*| (regret minus d/2 ln n) for n = 10^4, d = 1.
    # The discretized bound is an upper bound that depends on the binary expansion of theta* (a dyadic theta* such as 1 or 10 is cheap),
    # so the curve shows its WORST value over the window [theta, 1.05 theta] (200 equally spaced values).
    n = 10 ** 4
    ths = np.logspace(-1, 3, 81)


    def win_bound(t, n=n, width=1.05, pts=5000):
        vals = np.array([universal_bound_1d(n, x)[0] for x in np.linspace(t, width * t, pts)])
        return float(vals.max() - 0.5 * math.log(n)), float(np.median(vals) - 0.5 * math.log(n)), float(vals.min() - 0.5 * math.log(n))


    g1 = np.array([regret_exact(n, 1, t * t, 1.0) - 0.5 * math.log(n) for t in ths])
    g10 = np.array([regret_exact(n, 1, t * t, 100.0) - 0.5 * math.log(n) for t in ths])
    wb = np.array([win_bound(t) for t in ths])
    save("L5_prior_cost.dat", "theta gauss1 gauss10 universal_max universal_med", [ths, g1, g10, wb[:, 0], wb[:, 1]])
    for t in (1.0, 10.0, 100.0, 1000.0):
        mx, md, mn = win_bound(t)
        INFO["B3_t%g" % t] = (float(regret_exact(n, 1, t * t, 1.0) - 0.5 * math.log(n)),
                              float(regret_exact(n, 1, t * t, 100.0) - 0.5 * math.log(n)), mx, md, mn)
        print("   n=1e4, |theta*| in [%g, %g]: Gaussian(s=1) %.2f   Gaussian(s=10) %.2f   Elias-delta bound: max %.2f median %.2f min %.2f"
              "   [nats above (1/2) ln n]" % ((t, 1.05 * t) + INFO["B3_t%g" % t]))
        assert mx < 2 * math.log(max(t, 2.0)) + 12          # the universal charge is logarithmic in |theta*| with a modest constant
        assert mx > math.log(t) - 1 if t > 1 else True

    # ----------------------------------------------------------------------------------------------
    # C. MAP and the sequential plug-in predictor in the Gaussian example
    # ----------------------------------------------------------------------------------------------
    X = rng.standard_normal((9, 3)) + 1.0
    n, d = X.shape
    res = optimize.minimize(lambda t: 0.5 * np.sum((X - t) ** 2) + 0.5 * np.sum(t ** 2), np.zeros(d), tol=1e-14)
    assert np.max(np.abs(res.x - n * X.mean(axis=0) / (n + 1))) < 1e-6
    print("C1  MAP of NLL + |theta|^2/2 equals n xbar/(n+1)")


    def plugin_step(t, d, th2):          # E[ -log N(x_t; thetahat_{t-1}, I) + log N(x_t; theta*, I) ], thetahat = (t-1)/t xbar_{t-1}
        return 0.5 * ((t - 1) * d + th2) / t ** 2


    # Monte Carlo of the per-step expectation (t = 1, 2, 5, 30) in d = 2
    d, th = 2, np.array([1.3, -0.4])
    th2 = float(th @ th)
    for t in (1, 2, 5, 30):
        M = 400000
        prev = th + rng.standard_normal((M, t - 1, d)) if t > 1 else np.zeros((M, 0, d))
        thetahat = (t - 1) / t * prev.mean(axis=1) if t > 1 else np.zeros((M, d))
        x = th + rng.standard_normal((M, d))
        v = 0.5 * np.sum((x - thetahat) ** 2, axis=1) - 0.5 * np.sum((x - th) ** 2, axis=1)
        assert abs(v.mean() - plugin_step(t, d, th2)) < 6 * v.std() / math.sqrt(M), t
    print("C2  per-step regret of the MAP plug-in predictor, 1/2((t-1)d + |th*|^2)/t^2, confirmed by Monte Carlo")


    def plugin_regret(n, d, th2):
        t = np.arange(1, n + 1, dtype=float)
        return float(np.sum(0.5 * ((t - 1) * d + th2) / t ** 2))


    gam = np.euler_gamma
    for d, th2 in [(1, 0.0), (1, 4.0), (5, 30.0)]:
        n = 10 ** 6
        diff = plugin_regret(n, d, th2) - regret_exact(n, d, th2)
        lim = 0.5 * d * (gam + 1 - math.pi ** 2 / 6) + th2 * (math.pi ** 2 / 12 - 0.5)
        assert abs(diff - lim) < 1e-4, (d, th2, diff, lim)
    INFO["C_const_d"] = 0.5 * (gam + 1 - math.pi ** 2 / 6)
    INFO["C_const_th2"] = math.pi ** 2 / 12 - 0.5
    print("C3  plug-in minus Bayes regret -> d/2 (gamma + 1 - pi^2/6) + |th*|^2 (pi^2/12 - 1/2) = %.4f d + %.4f |th*|^2"
          % (INFO["C_const_d"], INFO["C_const_th2"]))
    INFO["C_plugin_n100_d1_th0"] = plugin_regret(100, 1, 0.0)
    INFO["C_bayes_n100_d1_th0"] = float(regret_exact(100, 1, 0.0))

    # ----------------------------------------------------------------------------------------------
    # D. Langevin dynamics
    # ----------------------------------------------------------------------------------------------
    try:
        import sympy as sp
        th_ = sp.symbols("theta", real=True)
        U = sp.Function("U")(th_)
        rho = sp.exp(-U)
        J = -(rho * sp.diff(U, th_) + sp.diff(rho, th_))             # probability current of  d theta = -U' dt + sqrt2 dW
        assert sp.simplify(J) == 0
        print("D1  (sympy) rho = exp(-U) has zero probability current, hence is stationary for d theta = -U' dt + sqrt(2) dW")
    except ImportError:
        print("D1  sympy not available: skipped (the identity J = -(rho U' + rho') = 0 for rho = e^-U is a one-line computation)")

    # stationary variance of  theta' = theta - eps(a theta + zeta) + sqrt(2 eps) xi  (U = a theta^2/2, mean zero):
    #   Var = (2 eps + eps^2 s^2) / (1 - (1 - eps a)^2) = (1 + eps s^2/2) / (a (1 - eps a/2))        [s^2 = variance of the gradient noise zeta]
    def ar1_var(eps, a, s2, inj=1.0):
        r = 1 - eps * a
        return (inj * 2 * eps + eps * eps * s2) / (1 - r * r)


    def closed_var(eps, a, s2, inj=1.0):
        return (inj + eps * s2 / 2) / (a * (1 - eps * a / 2))


    for eps, a, s2, inj in [(0.002, 21.0, 0.0, 1.0), (0.01, 21.0, 0.0, 1.0), (0.01, 21.0, 4.0, 1.0), (0.01, 21.0, 4.0, 0.0), (0.05, 3.0, 9.0, 0.0)]:
        assert abs(ar1_var(eps, a, s2, inj) - closed_var(eps, a, s2, inj)) < 1e-12 * closed_var(eps, a, s2, inj)
    # simulate many chains
    r = np.random.default_rng(11)
    for eps, a, s2, inj in [(0.002, 21.0, 0.0, 1.0), (0.01, 21.0, 0.0, 1.0), (0.05, 21.0, 0.0, 1.0), (0.01, 21.0, 4.0, 1.0), (0.01, 21.0, 4.0, 0.0)]:
        C, T = 40000, int(12 / (eps * a)) + 400
        th = np.zeros(C)
        for _ in range(T):
            th = th - eps * (a * th + math.sqrt(s2) * r.standard_normal(C)) + math.sqrt(2 * eps * inj) * r.standard_normal(C)
        est, pred = th.var(), closed_var(eps, a, s2, inj)
        assert abs(est / pred - 1) < 5 * math.sqrt(2.0 / C), (eps, a, s2, inj, est, pred)
    print("D2  stationary variance (1 + inj*..)/..: Euler-Maruyama 1/(a(1-eps a/2)), SGD eps s^2/(2a(1-eps a/2)), SGLD (1+eps s^2/2)/(a(1-eps a/2)) confirmed")
    INFO["D_var_ratio_a21_eps0.01"] = closed_var(0.01, 21.0, 0.0) * 21.0
    INFO["D_var_a21_eps0.01"] = closed_var(0.01, 21.0, 0.0)
    INFO["D_var_a21_eps0.002"] = closed_var(0.002, 21.0, 0.0)
    eps_1pct = 2 * (0.01 / 1.01) / 21.0
    assert abs(closed_var(eps_1pct, 21.0, 0.0) * 21.0 - 1.01) < 1e-12
    INFO["D_eps_for_1pct"] = eps_1pct
    print("   a=21: eps=0.01 -> variance %.5f (exact posterior 1/21 = %.5f, ratio %.4f);  eps=0.002 -> %.5f;  1%% bias needs eps = %.2e"
          % (INFO["D_var_a21_eps0.01"], 1 / 21, INFO["D_var_ratio_a21_eps0.01"], INFO["D_var_a21_eps0.002"], eps_1pct))

    # the numbers of the Langevin figure (gen_data.py, group 5): check the stored histogram against the exact posterior N(mu, 1/(n+1))
    dat = np.loadtxt(os.path.join(OUT, "langevin_hist.dat"), skiprows=1)
    info = dict(l.split() for l in open(os.path.join(OUT, "langevin_info.dat")))
    mu, sd = float(info["mu_post"]), float(info["sd_post"])
    assert abs(sd - 1 / math.sqrt(21)) < 1e-5
    dens = np.exp(-0.5 * ((dat[:, 0] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    assert np.max(np.abs(dens - dat[:, 2])) < 1e-4
    INFO["D_hist_maxdev_over_peak"] = float(np.max(np.abs(dat[:, 1] - dat[:, 2])) / dat[:, 2].max())
    print("D3  stored Langevin histogram: max deviation from the exact posterior density = %.1f%% of its peak" % (100 * INFO["D_hist_maxdev_over_peak"]))

    # ----------------------------------------------------------------------------------------------
    # E. Accuracy versus loss
    # ----------------------------------------------------------------------------------------------
    INFO["E_loss51"], INFO["E_loss99"], INFO["E_ln97"] = -math.log(0.51), -math.log(0.99), math.log(97)
    assert abs(INFO["E_loss51"] - 0.6733) < 1e-4 and abs(INFO["E_loss99"] - 0.01005) < 1e-5 and abs(INFO["E_ln97"] - 4.5747) < 1e-4
    # at 100% accuracy the loss can approach ln p: the correct class only needs to beat each other class
    p = 97
    for q in (1.0 / p + 1e-9, 0.51, 0.99):
        others = (1 - q) / (p - 1)
        assert q > others                                                   # correct class wins
    print("E   -ln0.51 = %.4f, -ln0.99 = %.5f, ln 97 = %.4f; correct class wins as soon as q > 1/p, so loss < ln p at perfect accuracy"
          % (INFO["E_loss51"], INFO["E_loss99"], INFO["E_ln97"]))
    INFO["E_loss_p7_51"] = -math.log(0.51)

    # ----------------------------------------------------------------------------------------------
    # F. Fourier identity and an explicit square-activation network for (a+b) mod p
    # ----------------------------------------------------------------------------------------------
    for p in (3, 5, 7, 11, 13, 23, 97):
        w = 2 * math.pi / p
        t = np.arange(-3 * p, 3 * p + 1)
        full = (1.0 / p) * sum(np.cos(w * k * t) for k in range(1, p))                # (1/p) sum_{k=1}^{p-1} cos(2 pi k t/p)
        assert np.max(np.abs(full - ((t % p == 0) - 1.0 / p))) < 1e-12
        if p % 2 == 1:
            half = (2.0 / p) * sum(np.cos(w * k * t) for k in range(1, (p - 1) // 2 + 1))
            assert np.max(np.abs(half - ((t % p == 0) - 1.0 / p))) < 1e-12
    # a single frequency already gives the right arg max when p is prime: cos(2 pi k t/p) = 1 only for t = 0 mod p
    for p in (3, 5, 7, 11, 13, 23, 97):
        mm = np.arange(p)
        for k in range(1, p):
            Fk = np.cos(2 * math.pi * k * ((mm[:, None, None] + mm[None, :, None] - mm[None, None, :]) % p) / p)     # [m, n, j]
            assert np.all(Fk.argmax(axis=2) == (mm[:, None] + mm[None, :]) % p), (p, k)
    print("F1  (1/p) sum_{k=1}^{p-1} cos(2 pi k t/p) = [t = 0 mod p] - 1/p  and the half sum 2/p sum_{k<=(p-1)/2} (p odd) verified")




    for p in (5, 7, 11, 13, 23):
        U, V, A = explicit_network(p)
        H = U.shape[0]
        assert H == 4 * (p - 1)
        hid = (U[:, :, None] + V[:, None, :]) ** 2                                     # H x p x p : h_i(m, n)
        F = np.einsum("ji,imn->jmn", A, hid)
        m_, n_ = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        target = np.zeros((p, p, p))
        for j in range(p):
            target[j] = ((m_ + n_) % p == j) - 1.0 / p
        assert np.max(np.abs(F - target)) < 1e-12
        assert np.all(F.argmax(axis=0) == (m_ + n_) % p)
        # every hidden unit has the same frequency in u and in v (the lecture's guess)
        def dom(Wm):
            spec = np.abs(np.fft.fft(Wm, axis=1))[:, 1:(p - 1) // 2 + 1]
            return spec.argmax(1) + 1
        assert np.all(dom(U) == dom(V))
    print("F2  explicit network: hidden width 4(p-1), f_j(m,n) = [j = m+n mod p] - 1/p to 1e-12 for p = 5, 7, 11, 13, 23; u and v share the frequency in every unit")
    INFO["F_width_p7"], INFO["F_width_p97"] = 4 * 6, 4 * 96

    # ----------------------------------------------------------------------------------------------
    # G. Neuron statistics from the stored table of grokking.py (seed 0)
    # ----------------------------------------------------------------------------------------------
    tab = np.loadtxt(os.path.join(OUT, "grokking_neurons.dat"), skiprows=1, dtype=int)
    assert tab.shape == (256, 4)
    half = 48
    for name, (ku, kv) in {"early": (tab[:, 0], tab[:, 1]), "final": (tab[:, 2], tab[:, 3])}.items():
        same = float(np.mean(ku == kv))
        pu = np.bincount(ku, minlength=half + 1)[1:] / len(ku)
        pv = np.bincount(kv, minlength=half + 1)[1:] / len(kv)
        chance_marg = float(np.sum(pu * pv))                                           # P(k_u = k_v) if u and v were independent with the same marginals
        rng2 = np.random.default_rng(3)
        perm = [float(np.mean(ku == rng2.permutation(kv))) for _ in range(5000)]
        INFO["G_%s_same" % name], INFO["G_%s_chance_marg" % name] = same, chance_marg
        INFO["G_%s_perm_max" % name] = float(np.max(perm))
        INFO["G_%s_distinct" % name] = int(len(set(ku)))
        print("G1  %-5s: same dominant frequency in u and v for %.3f of the 256 units; independent with the same marginals: %.3f "
              "(max over 5000 shuffles %.3f); distinct dominant frequencies of u: %d of %d" % (name, same, chance_marg, max(perm), len(set(ku)), half))
    assert abs(INFO["G_early_same"] - 0.363) < 0.001 and abs(INFO["G_final_same"] - 0.859) < 0.001

# ----------------------------------------------------------------------------------------------
# G2. (slow) What does the held-out part of the network compute?  Retrain the networks of figures/grokking.py (same seeds, same code path;
#     the accuracies are asserted to agree with figures/data/grokking_curves.dat) and, every 100 steps, look at the logits F_j(m,n) of ALL
#     pairs, centered over j.  For a set S of pairs (held-out pairs, or training pairs) define the "offset profile"
#            g_S(t) = mean over (m,n) in S of F_{m+n-t}(m,n)        (t = m+n-j mod p; sum_t g_S(t) = 0)
#     = the average logit of a class whose offset from the correct class is t.  The idealized Fourier-multiplication network has
#     F_j(m,n) = g(m+n-j) exactly, with g(0) = 1 - 1/p and g(t) = -1/p otherwise.  Quantities reported:
#       share_S  = (energy of F on S explained by its offset profile) / (energy of F on S)   = |S| sum_t g_S(t)^2 / sum_S sum_j F^2
#       margin   = g_S(0) p/(p-1)   (correct-class logit minus the mean of the others, on average),   noise = rms of F - g_S(m+n-j)
#       snr      = margin / noise;   acc_model = P(correct > all p-1 others) if the residuals were independent N(0, noise^2)
#     and, at the end of training, the Fourier spectrum of g_held-out (how many frequencies carry the generalizing part).
#     NOTE: the profile over ALL pairs is contaminated by the memorized training pairs (their correct-class logits are large), which is
#     why everything below is computed on the held-out pairs (and, for comparison, on the training pairs).
# ----------------------------------------------------------------------------------------------
if QUICK:
    print("--quick: skipping the PyTorch part")
else:
    import importlib.util
    import time
    import torch

    spec = importlib.util.spec_from_file_location("grokking_L5src", os.path.join(HERE, "grokking.py"))
    gk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gk)                              # only definitions; the experiment sits under __main__
    P, H, LR, WD, INIT, STEPS = gk.P, gk.H, gk.LR, gk.WD, gk.INIT, gk.STEPS
    SCRATCH = os.environ.get("L5_SCRATCH")                   # optional directory for weight snapshots (not part of the project)

    def logits_grid(W1, W2):
        """F[j, m, n] = sum_i W2[j, i] (W1[i, m] + W1[i, P + n])^2, centered over j (the softmax does not see the mean over j)."""
        u, v = W1[:, :P], W1[:, P:]
        hid = ((u[:, :, None] + v[:, None, :]) ** 2).reshape(W1.shape[0], P * P)
        F = (W2 @ hid).reshape(P, P, P)
        return F - F.mean(axis=0, keepdims=True)

    _z = np.linspace(-12, 12, 4801)
    _phi = np.exp(-0.5 * _z ** 2) / math.sqrt(2 * math.pi)

    def acc_model(snr):
        """P(correct class has the largest logit) if the correct logit is snr + N(0,1) and the other P-1 logits are iid N(0,1)."""
        cdf = 0.5 * (1.0 + special.erf((_z + snr) / math.sqrt(2.0)))
        return float(np.sum(_phi * cdf ** (P - 1)) * (_z[1] - _z[0]))

    def profile_stats(F, pairs):
        m_, n_ = pairs // P, pairs % P
        S = len(pairs)
        Fs = F[:, m_, n_]                                    # P x S  (class, pair)
        s_sum = (m_ + n_)
        cols = np.arange(S)
        g = np.array([Fs[(s_sum - t) % P, cols].mean() for t in range(P)])
        tmat = (s_sum[None, :] - np.arange(P)[:, None]) % P   # offset t = m+n-j of every (class, pair)
        inv = g[tmat]
        e_tot = float(np.sum(Fs ** 2))
        share = float(np.sum(inv ** 2) / e_tot)
        noise = float(np.sqrt(np.mean((Fs - inv) ** 2)))
        margin = float(g[0] * P / (P - 1))
        return {"share": share, "margin": margin, "noise": noise, "snr": margin / noise, "g": g}

    def spectrum_shares(g):
        """energy share of the frequencies k = 1..(P-1)/2 in the profile g (k and P-k folded)."""
        G = np.abs(np.fft.fft(g)) ** 2
        half = (P - 1) // 2
        e = G[1:half + 1] + G[P - 1:P - half - 1:-1]
        return e / e.sum()

    # sanity: the exact explicit network of part F has share 1, zero noise and (on any set of pairs) a perfect profile
    Ue, Ve, Ae = explicit_network(P)
    Fe = logits_grid(np.hstack([Ue, Ve]), Ae)
    rs = np.random.default_rng(0).permutation(P * P)[:3000]
    st = profile_stats(Fe, rs)
    assert st["share"] > 1 - 1e-9 and st["noise"] < 1e-9 and abs(st["g"][0] - (1 - 1.0 / P)) < 1e-9
    sp = spectrum_shares(st["g"])
    assert np.max(np.abs(sp - 1.0 / len(sp))) < 1e-9              # the ideal profile has equal energy at every frequency
    assert abs(acc_model(0.0) - 1.0 / P) < 1e-4 and acc_model(9.0) > 0.99999
    print("G2  explicit network (p = 97): offset-profile share %.12f, noise %.1e; ideal profile has equal energy at all 48 frequencies" % (st["share"], st["noise"]))

    def run_seed(seed, every=100):
        g, X, y, tr, te = gk.make_data(seed)
        W1 = torch.nn.Parameter(INIT * (torch.rand(H, 2 * P, generator=g) * 2 - 1) / math.sqrt(2 * P))
        W2 = torch.nn.Parameter(INIT * (torch.rand(P, H, generator=g) * 2 - 1) / math.sqrt(H))
        opt = torch.optim.AdamW([W1, W2], lr=LR, weight_decay=WD, betas=(0.9, 0.98))
        te_np, tr_np = te.numpy(), tr.numpy()
        rec, snaps = [], {}
        for step in range(STEPS + 1):
            logits = ((X[tr] @ W1.T) ** 2) @ W2.T
            loss = torch.nn.functional.cross_entropy(logits, y[tr])
            if step % every == 0:
                with torch.no_grad():
                    lt = ((X[te] @ W1.T) ** 2) @ W2.T
                    tr_acc = (logits.argmax(1) == y[tr]).float().mean().item()
                    te_acc = (lt.argmax(1) == y[te]).float().mean().item()
                w1n, w2n = W1.detach().numpy().astype(np.float64), W2.detach().numpy().astype(np.float64)
                F = logits_grid(w1n, w2n)
                a, b = profile_stats(F, te_np), profile_stats(F, tr_np)
                rec.append((step, tr_acc, te_acc, a["share"], b["share"], a["snr"], b["snr"], a["margin"], a["noise"], acc_model(a["snr"])))
                last = (a, b)
                if SCRATCH and step in (240, 1000, 3000, 5000, 10000):
                    snaps["W1_%d" % step], snaps["W2_%d" % step] = w1n, w2n
            opt.zero_grad()
            loss.backward()
            opt.step()
        if SCRATCH and snaps:
            np.savez(os.path.join(SCRATCH, "L5_seed%d_snapshots.npz" % seed), **snaps)
        return np.array(rec), last

    results = []
    for s_ in range(4):
        t0 = time.time()
        results.append(run_seed(s_))
        print("   seed %d trained and analyzed in %.0f s" % (s_, time.time() - t0), flush=True)
    runs = [r for r, _ in results]
    # consistency with figures/data/grokking_curves.dat (seed 0): same accuracies at the common steps (every 100 steps)
    ref = np.loadtxt(os.path.join(OUT, "grokking_curves.dat"), skiprows=1)
    ref = {int(r[0]): r for r in ref}
    for row in runs[0]:
        r_ = ref[int(row[0])]
        assert abs(row[1] - r_[1]) < 1e-6 and abs(row[2] - r_[2]) < 1e-6, row
    steps = runs[0][:, 0]
    # columns of rec: 0 step, 1 train acc, 2 test acc, 3 share(test), 4 share(train), 5 snr(test), 6 snr(train), 7 margin(test), 8 noise(test), 9 acc_model(snr(test))
    cols = [steps] + [r[:, 2] for r in runs] + [r[:, 3] for r in runs] + [r[:, 5] for r in runs] + [r[:, 9] for r in runs] + [r[:, 4] for r in runs]
    save("L5_grok_profile.dat", "step test0 test1 test2 test3 share0 share1 share2 share3 snr0 snr1 snr2 snr3 model0 model1 model2 model3 trshare0 trshare1 trshare2 trshare3", cols)
    for s_, (r, (a, b)) in enumerate(results):
        i_fit = int(np.argmax(r[:, 1] >= 0.99))
        i99 = int(np.argmax(r[:, 2] >= 0.99))
        sh = spectrum_shares(a["g"])
        order = np.argsort(sh)[::-1]
        cum = np.cumsum(sh[order])
        k90 = int(np.argmax(cum >= 0.90) + 1)
        print("   seed %d: first snapshot with test acc >= 99%%: step %d ; with train acc >= 99%%: step %d" % (s_, r[i99, 0], r[i_fit, 0]))
        print("      held-out share of the offset profile: step 0 %.5f | train fit %.4f | 1000: %.4f | 2000: %.4f | 3000: %.4f | 5000: %.4f | final: %.4f"
              % (r[0, 3], r[i_fit, 3], r[10, 3], r[20, 3], r[30, 3], r[50, 3], r[-1, 3]))
        print("      training-pair share: train fit %.4f | 1000: %.4f | 3000: %.4f | final: %.4f" % (r[i_fit, 4], r[10, 4], r[30, 4], r[-1, 4]))
        print("      held-out snr (margin/noise): 500: %.3f | 1000: %.3f | 2000: %.3f | 3000: %.3f | 4000: %.3f | 5000: %.3f | final: %.3f"
              % (r[5, 5], r[10, 5], r[20, 5], r[30, 5], r[40, 5], r[50, 5], r[-1, 5]))
        print("      held-out acc (actual | gaussian model of snr): 1000: %.3f | %.3f ; 2000: %.3f | %.3f ; 3000: %.3f | %.3f ; 4000: %.3f | %.3f ; 5000: %.3f | %.3f"
              % (r[10, 2], r[10, 9], r[20, 2], r[20, 9], r[30, 2], r[30, 9], r[40, 2], r[40, 9], r[50, 2], r[50, 9]))
        print("      final profile spectrum: top-5 frequencies %s carry %.3f of the energy; frequencies needed for 90%%: %d of 48; largest share %.3f"
              % (list(order[:5] + 1), cum[4], k90, sh[order[0]]))
        for k_, v_ in (("share_fit", r[i_fit, 3]), ("share_1000", r[10, 3]), ("share_2000", r[20, 3]), ("share_3000", r[30, 3]), ("share_5000", r[50, 3]),
                       ("share_final", r[-1, 3]), ("trshare_final", r[-1, 4]), ("trshare_fit", r[i_fit, 4]), ("snr_500", r[5, 5]), ("snr_1000", r[10, 5]),
                       ("snr_3000", r[30, 5]), ("snr_5000", r[50, 5]), ("snr_final", r[-1, 5]), ("t99", r[i99, 0]), ("k90", k90), ("top5", cum[4]),
                       ("top1", sh[order[0]])):
            INFO["G2_seed%d_%s" % (s_, k_)] = float(v_)
        if s_ == 0:
            np.savetxt(os.path.join(OUT, "L5_profile_spectrum.dat"), np.column_stack([np.arange(1, 49), sh]), header="k share", comments="", fmt="%.8e")

with open(os.path.join(OUT, "L5_info.dat"), "w") as fh:
    for k, v in INFO.items():
        fh.write("%s %s\n" % (k, v))
print("wrote figures/data/L5_info.dat")
print("ALL CHECKS PASSED")
