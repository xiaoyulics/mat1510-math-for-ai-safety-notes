#!/usr/bin/env python3
"""
Lecture 7 -- numerical checks and figure data for the persona-mixture toy model.

Run:   /opt/anaconda3/bin/python3 -I figures/gen_data_L7.py        (numpy + scipy, about 1-2 minutes)

Writes figures/data/L7_*.dat (whitespace separated, one header line) and asserts every closed form
that the notes state against an exact or numerical computation.  Everything is deterministic
(fixed seeds; the only Monte Carlo is a sanity check of a variance).

Model (see the notes, Section 7.8 and the 'rigorous version' boxes)
    persona z in M, prior pi0(z), context likelihood p(c|z), badness U(z) in [0,1], potential V(z),
    posterior  pi(z|c) ~ pi0(z) p(c|z),   tilted posterior  pi_beta(z|c) ~ pi(z|c) exp(-beta V(z)),
    P_beta(c) = E_{pi_beta(.|c)}[U].

Blocks
    A  finite persona set: tilt calculus (derivative = -Cov, second derivative = third cumulant,
       Gibbs variational principle, KL budget)
    B  two personas with Bernoulli tokens: exact expected bad-probability, exact first-passage time
       (hitting-time theorem), Wald bounds, linear delay in beta
    C  regular manifold d=1,2 (adaptive-grid exact posteriors), lattice of spacing h (crossover),
       finite persona set (exponential decay), average-versus-typical (Bhattacharyya rate)
    D  singular family: slope -1/3
    E  an atom inside a continuum: power law n^{-d/2}
    F  numbers for the exercises (educational context, inoculation, closed forms)
"""
import os
import numpy as np
from scipy.special import expit, logsumexp, log_softmax, softmax, gammaln
from scipy.stats import binom, norm
from scipy.optimize import minimize
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def write_table(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(("nan" if (isinstance(x, float) and not np.isfinite(x)) else f"{x:.10g}") for x in r) + "\n")
    print("  wrote", os.path.relpath(path, HERE))


def banner(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# =====================================================================================================
# A. finite persona set: the calculus of the tilt
# =====================================================================================================
def block_A():
    banner("A. tilt calculus on a finite persona set")
    rng = np.random.default_rng(7)
    K = 6
    pi = softmax(rng.normal(size=K))          # posterior given the context, no tilt
    U = rng.random(K)
    V = 2.0 * rng.normal(size=K)

    def tilt(beta):
        w = pi * np.exp(-beta * V)
        return w / w.sum()

    def P(beta):
        return tilt(beta) @ U

    def cov(x, y, q):
        return q @ (x * y) - (q @ x) * (q @ y)

    def kappa3(x, y, z, q):                    # joint third cumulant
        mx, my, mz = q @ x, q @ y, q @ z
        return q @ ((x - mx) * (y - my) * (z - mz))

    # (b) d/dbeta P = -Cov_beta(U,V),  d^2/dbeta^2 P = kappa_beta(U,V,V)
    for beta in [0.0, 0.7, 2.0]:
        h = 1e-4
        d1 = (P(beta + h) - P(beta - h)) / (2 * h)
        q = tilt(beta)
        assert abs(d1 + cov(U, V, q)) < 1e-8, (d1, cov(U, V, q))
        d2 = (P(beta + h) - 2 * P(beta) + P(beta - h)) / h ** 2
        assert abs(d2 - kappa3(U, V, V, q)) < 1e-5, (d2, kappa3(U, V, V, q))
    print("  dP/dbeta = -Cov_beta(U,V) and d2P/dbeta2 = kappa_beta(U,V,V): verified at beta = 0, 0.7, 2")

    # (c) Gibbs variational principle: pi_beta minimizes beta E_q[V] + KL(q||pi)
    beta = 1.3

    def objective(theta):
        q = softmax(theta)
        return beta * (q @ V) + np.sum(q * (np.log(q) - np.log(pi)))

    res = minimize(objective, np.zeros(K), method="BFGS", options={"gtol": 1e-12})
    assert np.max(np.abs(softmax(res.x) - tilt(beta))) < 1e-6
    Z = np.sum(pi * np.exp(-beta * V))
    assert abs(res.fun - (-np.log(Z))) < 1e-9
    print("  Gibbs variational principle: minimizer = tilted posterior, minimum value = -log Z  (verified)")

    # (d) KL budget:  P_beta - P_0 ~ -sqrt(2 KL) sd0(U) corr0(U,V)
    q0 = tilt(0.0)
    sdU = np.sqrt(cov(U, U, q0))
    sdV = np.sqrt(cov(V, V, q0))
    corr = cov(U, V, q0) / (sdU * sdV)
    for beta in [1e-2, 1e-3]:
        q = tilt(beta)
        KL = np.sum(q * (np.log(q) - np.log(q0)))
        lhs = P(beta) - P(0.0)
        rhs = -np.sign(beta) * np.sqrt(2 * KL) * sdU * corr
        assert abs(lhs / rhs - 1) < 20 * beta, (beta, lhs, rhs)
    print("  KL budget  dP ~ -sqrt(2KL) sd(U) corr(U,V): verified (relative error O(beta))")

    # beta -> infinity: concentrates on argmin V
    assert abs(P(60.0) - U[np.argmin(V)]) < 1e-9
    print("  P_beta -> U(argmin V) as beta -> infinity: verified")
    return dict(corr=corr)


# =====================================================================================================
# B. two personas, Bernoulli tokens
# =====================================================================================================
def block_B():
    banner("B. two personas A (safe) and B (unsafe): exact expected bad-probability and crossing times")
    pA, pB = 1 / 3, 2 / 3
    ln2 = np.log(2.0)
    ell1, ell0 = np.log(pB / pA), np.log((1 - pB) / (1 - pA))
    assert abs(ell1 - ln2) < 1e-15 and abs(ell0 + ln2) < 1e-15
    D = pB * ell1 + (1 - pB) * ell0
    assert abs(D - ln2 / 3) < 1e-14                       # KL(B||A) = (ln 2)/3 nats per token
    L0 = np.log(0.1 / 0.9)                                # prior P(B) = 0.1
    dV = 1.0
    M = ln2                                              # largest increment

    def meanP(n, beta):
        k = np.arange(n + 1)
        logw = binom.logpmf(k, n, pB)
        L = L0 - beta * dV + k * ell1 + (n - k) * ell0
        return float(np.sum(np.exp(logw) * expit(L)))

    betas_curves = [0, 4, 8, 12, 16]
    ns = np.arange(0, 121)
    curves = {b: np.array([meanP(n, b) for n in ns]) for b in betas_curves}
    write_table("L7_delay_curves.dat", ["n"] + [f"P{b}" for b in betas_curves],
                [[int(n)] + [curves[b][i] for b in betas_curves] for i, n in enumerate(ns)])

    # exact first-passage law of the +-1 walk (hitting-time theorem): P(T=t) = (m/t) C(t,(t+m)/2) p^{(t+m)/2} q^{(t-m)/2}
    p, q = pB, 1 - pB

    def passage_moments(m, tmax=6000):
        t = np.arange(m, tmax + 1, 2)
        a = (t + m) // 2
        logp = np.log(m / t) + gammaln(t + 1) - gammaln(a + 1) - gammaln(t - a + 1) + a * np.log(p) + (t - a) * np.log(q)
        pr = np.exp(logp)
        assert abs(pr.sum() - 1) < 1e-9
        mean = np.sum(t * pr)
        var = np.sum(t ** 2 * pr) - mean ** 2
        return mean, var

    rows = []
    max_dev = 0.0
    for beta in np.arange(0, 21, 1.0):
        h = beta * dV - L0
        m = int(np.floor(h / ln2)) + 1                    # first level strictly above the threshold h
        ET, VT = passage_moments(m)
        assert abs(ET - m / (pB - q)) < 1e-6 and abs(VT - m * (1 - (pB - q) ** 2) / (pB - q) ** 3) < 1e-5
        lo, hi = h / D, (h + M) / D                       # Wald bounds:  h/D <= E T <= (h+M)/D
        assert lo - 1e-9 <= ET <= hi + 1e-9, (beta, lo, ET, hi)
        # crossing of the mean curve (50 percent), interpolated
        nn = np.arange(0, 400)
        PP = np.array([meanP(n, beta) for n in nn])
        i = int(np.argmax(PP >= 0.5))
        nstar = nn[i - 1] + (0.5 - PP[i - 1]) / (PP[i] - PP[i - 1]) if i > 0 else 0.0
        rows.append([beta, nstar, ET, lo, hi, VT])
        max_dev = max(max_dev, abs(nstar - lo))
    write_table("L7_delay_cross.dat", ["beta", "nstar_curve", "ET_exact", "lin", "upper", "VarT"], rows)
    print(f"  D = KL(B||A) = (ln 2)/3 = {D:.6f};  E[T] lies in [h/D, (h+ln2)/D] for beta = 0..20 (exact, hitting-time theorem)")
    print(f"  mean-curve crossing differs from h/D by at most {max_dev:.2f} tokens (beta = 0..20)")
    assert max_dev < 1.0
    # the delay is linear in beta up to the bounded error M/D
    E = np.array([r[2] for r in rows])
    betas = np.array([r[0] for r in rows])
    slope = np.polyfit(betas, E, 1)[0]
    print(f"  slope of E[T_beta] against beta: {slope:.4f}  (prediction Delta V / D = {dV / D:.4f})")
    assert abs(slope - dV / D) < 0.05
    assert np.all(np.abs((E[1:] - E[0]) - betas[1:] * dV / D) <= M / D + 1e-9)

    # Monte Carlo check of the CLT variance  Var T ~ sigma^2 h / D^3
    rng = np.random.default_rng(3)
    N = 40000
    beta = 16.0
    h = beta * dV - L0
    m = int(np.floor(h / ln2)) + 1
    pos = np.zeros(N, int)
    T = np.zeros(N, int)
    alive = np.ones(N, bool)
    while alive.any():
        step = np.where(rng.random(N) < pB, 1, -1)
        pos[alive] += step[alive]
        T[alive] += 1
        alive &= pos < m
    sig2 = pB * ell1 ** 2 + (1 - pB) * ell0 ** 2 - D ** 2
    print(f"  Monte Carlo (beta=16): E T = {T.mean():.2f}, Var T = {T.var():.0f};  exact {m / (pB - q):.2f}, {m * (1 - (pB - q) ** 2) / (pB - q) ** 3:.0f};"
          f"  sigma^2 h/D^3 = {sig2 * h / D ** 3:.0f}")
    assert abs(T.mean() - m / (pB - q)) < 0.5

    return dict(D=D, L0=L0)


# =====================================================================================================
# C. manifolds of personas: regular d = 1, 2; lattices; finite sets; average versus typical
# =====================================================================================================
class SoftmaxFamily:
    """Persona z in R^d; tokens are categorical with logits W z.  'Typical context' of length n drawn from persona z*:
    the log-likelihood is n * sum_v p*(v) log p_z(v) (the expected log-likelihood), so that the posterior is a deterministic
    function of n (no sampling noise)."""

    def __init__(self, W, zstar, a, u_target, w, nu=3.0):
        self.W = np.asarray(W, float)
        self.zs = np.asarray(zstar, float)
        self.d = len(self.zs)
        self.ps = np.exp(self.logp(self.zs[None, :])[0])
        self.I = self.W.T @ (np.diag(self.ps) - np.outer(self.ps, self.ps)) @ self.W
        self.nu = nu
        self.a = np.asarray(a, float)
        self.u0 = np.log(u_target / (1 - u_target)) - self.a @ self.zs
        self.w = np.asarray(w, float)

    def logp(self, z):
        return log_softmax(z @ self.W.T, axis=-1)

    def loglik(self, z, n):
        return n * (self.ps[None, :] * self.logp(z)).sum(1)

    def logprior(self, z):
        return -(self.nu + self.d) / 2 * np.log1p((z ** 2).sum(1) / self.nu)

    def U(self, z):
        return expit(z @ self.a + self.u0)

    def V(self, z):
        return np.logaddexp(0, z @ self.w)

    def gradU(self, z):
        u = expit(z @ self.a + self.u0)
        return u * (1 - u) * self.a

    def gradV(self, z):
        return expit(z @ self.w) * self.w

    def coef(self, beta):                  # predicted limit of n (P_beta - P_0)
        return -beta * self.gradU(self.zs) @ np.linalg.solve(self.I, self.gradV(self.zs))

    def kl_to(self, z):                    # KL(p_{z*} || p_z) for rows z
        return (self.ps[None, :] * (np.log(self.ps)[None, :] - self.logp(z))).sum(1)

    def dP_adaptive(self, n, beta, T=9.0, N=None):
        lam, Q = np.linalg.eigh(self.I)
        if self.d == 1:
            N = N or 4001
            t = np.linspace(-T, T, N)
            z = self.zs + (t / np.sqrt(n * lam[0]))[:, None] * Q[:, 0][None, :]
        else:
            N = N or 361
            t = np.linspace(-T, T, N)
            T1, T2 = np.meshgrid(t, t, indexing="ij")
            z = (self.zs[None, None, :] + (T1 / np.sqrt(n * lam[0]))[..., None] * Q[:, 0]
                 + (T2 / np.sqrt(n * lam[1]))[..., None] * Q[:, 1]).reshape(-1, 2)
        ll = self.loglik(z, n)
        out = []
        for b in (0.0, beta):
            lw = self.logprior(z) + ll - b * self.V(z)
            wt = np.exp(lw - lw.max())
            wt /= wt.sum()
            out.append(float(wt @ self.U(z)))
        return out[0], out[1]

    def lattice_dP(self, n, beta, pts):
        """pts: array of persona positions (K,d) with pts[0] == z*  (so r_k = w_k / w_*).  Returns P_beta - P_0 computed relative
        to the dominant persona z*, free of cancellation."""
        ll = self.loglik(pts, n)
        res = []
        for b in (0.0, beta):
            lw = self.logprior(pts) + ll - b * self.V(pts)
            r = np.exp(lw[1:] - lw[0])
            dev = (r * (self.U(pts[1:]) - self.U(pts[:1]))).sum() / (1.0 + r.sum())
            res.append(dev)
        return res[1] - res[0]


def block_C():
    banner("C. manifolds of personas (d=1,2), lattices of spacing h, finite sets")
    beta = 2.0
    fam1 = SoftmaxFamily(W=[[1.0], [0.4], [-0.3], [-1.1]], zstar=[0.3], a=[1.2], u_target=0.2, w=[1.5])
    fam2 = SoftmaxFamily(W=[[1.0, 0.2], [0.5, -0.8], [-0.4, 0.9], [-1.1, -0.3]], zstar=[0.3, -0.2],
                         a=[1.0, 0.7], u_target=0.2, w=[0.9, 1.4])
    c1, c2 = fam1.coef(beta), fam2.coef(beta)
    print(f"  Fisher information d=1: {fam1.I[0, 0]:.6f};  d=2: eigenvalues {np.linalg.eigvalsh(fam2.I)}")
    print(f"  predicted limits  n*(P_beta-P_0) -> {c1:.6f} (d=1),  {c2:.6f} (d=2)")
    assert c1 < 0 and c2 < 0

    # grid-convergence check at small and moderate n
    for fam, n in [(fam1, 10), (fam1, 100), (fam2, 10), (fam2, 100)]:
        a1 = fam.dP_adaptive(n, beta)
        a2 = fam.dP_adaptive(n, beta, T=12.0, N=(6001 if fam.d == 1 else 541))
        assert abs((a1[1] - a1[0]) - (a2[1] - a2[0])) < 1e-9, (n, a1, a2)
    print("  adaptive grids converged (window +-9 sigma versus +-12 sigma, finer mesh): differences < 1e-9")

    ns = np.logspace(1, 5, 41)
    rows = []
    d1 = []
    d2 = []
    for n in ns:
        P0, Pb = fam1.dP_adaptive(n, beta)
        Q0, Qb = fam2.dP_adaptive(n, beta)
        d1.append(Pb - P0)
        d2.append(Qb - Q0)
    d1, d2 = np.array(d1), np.array(d2)
    r1, r2 = ns * d1 / c1, ns * d2 / c2
    print(f"  n (P_b - P_0) / limit  at n=1e2, 1e3, 1e4, 1e5:  d=1: {r1[[10, 20, 30, 40]]}   d=2: {r2[[10, 20, 30, 40]]}")
    assert abs(r1[-1] - 1) < 1e-3 and abs(r2[-1] - 1) < 1e-3 and abs(r1[30] - 1) < 2e-2 and abs(r2[30] - 1) < 2e-2
    sl1 = np.polyfit(np.log(ns[30:]), np.log(np.abs(d1[30:])), 1)[0]
    sl2 = np.polyfit(np.log(ns[30:]), np.log(np.abs(d2[30:])), 1)[0]
    print(f"  log-log slopes for n >= 1e4:  d=1: {sl1:.4f},  d=2: {sl2:.4f}   (prediction -1)")
    assert abs(sl1 + 1) < 5e-3 and abs(sl2 + 1) < 5e-3

    # random contexts (Monte Carlo over data) for d = 1: the mean over contexts of the difference also scales like 1/n
    rng = np.random.default_rng(11)
    zgrid = np.linspace(-3.0, 3.6, 3301)[:, None]
    logp_grid = fam1.logp(zgrid)
    lp0 = fam1.logprior(zgrid)
    Ug, Vg = fam1.U(zgrid), fam1.V(zgrid)
    mc_rows = []
    for n in [100, 400, 1600]:
        acc = []
        for _ in range(400):
            counts = rng.multinomial(n, fam1.ps)
            ll = logp_grid @ counts
            res = []
            for b in (0.0, beta):
                lw = lp0 + ll - b * Vg
                wt = np.exp(lw - lw.max())
                wt /= wt.sum()
                res.append(wt @ Ug)
            acc.append(res[1] - res[0])
        mc_rows.append((n, np.mean(acc), np.std(acc) / np.sqrt(len(acc))))
    for n, mval, se in mc_rows:
        print(f"  random contexts, d=1: n={n:5d}  n*mean(dP) = {n * mval:.4f} +- {n * se:.4f}   (typical-context limit {c1:.4f})")
        assert abs(n * mval - c1) < 5 * n * se + 0.03 * abs(c1)

    # ---- lattices of personas with spacing h (z* sits on the lattice) -------------------------------------------------
    lat_rows = []
    hs = [0.4, 0.2, 0.1]
    lattice_curves = {}
    for h in hs:
        K = int(min(1500, 60 / h))
        pts = fam1.zs[None, :] + h * np.arange(0, K + 1)[:, None] * 1.0
        # build both sides, z* first
        left = fam1.zs[None, :] - h * np.arange(1, K + 1)[:, None]
        pts = np.vstack([fam1.zs[None, :], pts[1:], left])
        vals = np.array([fam1.lattice_dP(n, beta, pts) for n in ns])
        lattice_curves[h] = vals
    # crossover n_half: first n at which |dP_lattice| < 0.5 |dP_continuum|  ;  predicted n_half * I h^2 = const
    consts = []
    for h in hs:
        ratio = np.abs(lattice_curves[h]) / np.abs(d1)
        # use a fine log grid for the crossover
        nf = np.logspace(1, 5, 401)
        pts = np.vstack([fam1.zs[None, :], fam1.zs[None, :] + h * np.arange(1, int(min(1500, 60 / h)) + 1)[:, None],
                         fam1.zs[None, :] - h * np.arange(1, int(min(1500, 60 / h)) + 1)[:, None]])
        lat = np.array([abs(fam1.lattice_dP(n, beta, pts)) for n in nf])
        con = np.array([abs(x[1] - x[0]) for x in (fam1.dP_adaptive(n, beta) for n in nf)])
        idx = np.argmax(lat / con < 0.5)
        consts.append(nf[idx] * fam1.I[0, 0] * h * h)
        print(f"  lattice h={h}: |dP| drops below half of the continuum value at n = {nf[idx]:.0f};  n * I * h^2 = {consts[-1]:.2f}")
    assert max(consts) / min(consts) < 1.25 and 3.0 < np.mean(consts) < 8.0
    nhalf_const = float(np.mean(consts))

    # exponential regime of a lattice: slope of log|dP| -> -(smaller of the two nearest-neighbour KLs)
    h = 0.4
    pts = np.vstack([fam1.zs[None, :], fam1.zs[None, :] + h * np.arange(1, 160)[:, None], fam1.zs[None, :] - h * np.arange(1, 160)[:, None]])
    KLnn = min(fam1.kl_to(fam1.zs[None, :] + h)[0], fam1.kl_to(fam1.zs[None, :] - h)[0])
    n1, n2 = 1500.0, 2000.0
    s = (np.log(abs(fam1.lattice_dP(n2, beta, pts))) - np.log(abs(fam1.lattice_dP(n1, beta, pts)))) / (n2 - n1)
    print(f"  lattice h=0.4: exponential rate {-s:.5f} per token (n = 1500..2000); KL to the nearer neighbour {KLnn:.5f}; I h^2/2 = {fam1.I[0, 0] * h * h / 2:.5f}")
    assert abs(-s / KLnn - 1) < 5e-3

    # ---- a finite set of 5 personas ----------------------------------------------------------------------------------
    hd = 0.8
    disc = np.vstack([fam1.zs[None, :], fam1.zs[None, :] + hd * np.array([[1.0], [2.0], [-1.0], [-2.0]])])
    KLs = fam1.kl_to(disc[1:])
    Dmin = float(KLs.min())
    ndisc = np.arange(1, 121, 1.0)
    ddisc = np.array([fam1.lattice_dP(n, beta, disc) for n in ndisc])

    def first_order(n, idx):
        pr = np.exp(fam1.logprior(disc[1:][idx]) - fam1.logprior(disc[:1]))
        dU = fam1.U(disc[1:][idx]) - fam1.U(disc[:1])
        tl = np.exp(-beta * (fam1.V(disc[1:][idx]) - fam1.V(disc[:1]))) - 1.0
        return np.sum((pr * dU * tl)[None, :] * np.exp(-np.outer(np.atleast_1d(n), KLs[idx])), axis=1)

    allk = np.arange(len(KLs))
    near = np.array([int(np.argmin(KLs))])
    pred_all = first_order(ndisc, allk)
    pred_one = first_order(ndisc, near)
    r_all = ddisc[-1] / pred_all[-1]
    nbig = np.array([300.0, 400.0])
    exact_big = np.array([fam1.lattice_dP(n, beta, disc) for n in nbig])
    r_one = exact_big / first_order(nbig, near)
    print(f"  5 personas (spacing 0.8): D_min = {Dmin:.5f} per token, all four KLs {np.round(KLs, 4)}")
    print(f"       exact / first-order(all four) at n=120: {r_all:.5f};  exact / single exponential C e^(-D_min n) at n=300, 400: {r_one}")
    assert abs(r_all - 1) < 1e-3 and np.all(np.abs(r_one - 1) < 1e-3)
    write_table("L7_discrete.dat", ["n", "absdP", "asympt"],
                [[n, abs(a), abs(b)] for n, a, b in zip(ndisc, ddisc, pred_one) if abs(a) > 1e-13])

    # ---- singular family (D): one-dimensional, K(theta) = theta^6/2 ----------------------------------------------------
    sing = block_D(beta, ns)

    # write the main decay table
    def clip(x):
        return x if abs(x) > 1e-13 else float("nan")
    rows = []
    for i, n in enumerate(ns):
        rows.append([n, abs(d1[i]), abs(d2[i]), abs(sing["vals"][i]), clip(abs(lattice_curves[0.4][i])),
                     clip(abs(lattice_curves[0.2][i])), clip(abs(lattice_curves[0.1][i])),
                     abs(c1) / n, abs(c2) / n, abs(sing["coef"]) * n ** (-1 / 3)])
    write_table("L7_decay.dat", ["n", "reg1", "reg2", "sing", "lat04", "lat02", "lat01", "as1", "as2", "assing"], rows)
    with open(os.path.join(DATA, "L7_decay_info.dat"), "w") as f:
        f.write("c1 c2 csing I1 nhalfconst Dmin beta\n")
        f.write(f"{c1:.8g} {c2:.8g} {sing['coef']:.8g} {fam1.I[0, 0]:.8g} {nhalf_const:.6g} {Dmin:.8g} {beta}\n")
    print("  wrote figures/data/L7_decay_info.dat")

    # ---- average versus typical (Bhattacharyya) -------------------------------------------------------------------------
    block_C2()
    return dict(c1=c1, c2=c2, I1=fam1.I[0, 0], nhalf=nhalf_const, sing=sing["coef"])


def block_D(beta, ns):
    """tokens ~ N(theta^3, 1), truth theta* = 0, typical context xbar = 0: loglik = -(n/2) theta^6 (singular: Fisher info 9 theta^4 vanishes at 0)."""
    nu = 3.0
    a, w = 1.2, 1.5
    u0 = np.log(0.2 / 0.8)
    U = lambda t: expit(a * t + u0)
    V = lambda t: np.logaddexp(0, w * t)
    logprior = lambda t: -(nu + 1) / 2 * np.log1p(t ** 2 / nu)

    def dP(n, N=200001, T=8.0):
        s = np.linspace(-T, T, N)
        t = s * n ** (-1 / 6)
        out = []
        for bb in (0.0, beta):
            lw = logprior(t) - n / 2 * t ** 6 - bb * V(t)
            wt = np.exp(lw - lw.max())
            wt /= wt.sum()
            out.append(float(wt @ U(t)))
        return out[1] - out[0]

    vals = np.array([dP(n) for n in ns])
    m2 = quad(lambda s: s ** 2 * np.exp(-s ** 6 / 2), -10, 10)[0] / quad(lambda s: np.exp(-s ** 6 / 2), -10, 10)[0]
    m2_closed = 2 ** (1 / 3) * np.exp(gammaln(0.5) - gammaln(1 / 6))
    assert abs(m2 - m2_closed) < 1e-9
    Up, Vp = a * 0.2 * 0.8, w * 0.5
    coef = -beta * Up * Vp * m2
    # the corrections are of relative order n^(-1/3) for the symmetric prior used (n^(-1/6) in general), so the asymptotics is tested on an extended range
    ne = np.logspace(8, 14, 13)
    ve = np.array([dP(n) for n in ne])
    sl = np.polyfit(np.log(ne[6:]), np.log(np.abs(ve[6:])), 1)[0]
    sl_fig = np.polyfit(np.log(ns[30:]), np.log(np.abs(vals[30:])), 1)[0]
    print(f"  singular family N(theta^3,1): slope {sl:.5f} for n = 1e11..1e14 (prediction -1/3); {sl_fig:.4f} on the plotted range n = 1e4..1e5; "
          f"n^(1/3) dP -> {ve[-1] * ne[-1] ** (1 / 3):.6f} vs -beta U'(0) V'(0) m2 = {coef:.6f}, m2 = 2^(1/3) Gamma(1/2)/Gamma(1/6) = {m2_closed:.6f}")
    assert abs(sl + 1 / 3) < 2e-3
    assert abs(ve[-1] * ne[-1] ** (1 / 3) / coef - 1) < 1e-3
    return dict(vals=vals, coef=coef, m2=m2_closed)


def block_C2():
    """Average over contexts versus typical context for two Gaussian personas N(0,1) and N(Delta,1); context from B = N(Delta,1)."""
    Delta = 1.0
    out_rows = []
    gh_x = np.linspace(-14.0, 14.0, 56001)                # trapezoid rule for the standard normal weight
    gh_w = norm.pdf(gh_x)
    gh_w = gh_w / gh_w.sum()
    ns = np.arange(2, 121, 2)
    mean_err, typ_err = [], []
    for n in ns:
        # L_n = n Delta^2/2 + sqrt(n) Delta Z  (L0 = 0);  error = sigma(-L_n) = probability the posterior puts on the wrong persona A
        L = n * Delta ** 2 / 2 + np.sqrt(n) * Delta * gh_x
        mean_err.append(float(gh_w @ expit(-L)))
        typ_err.append(float(expit(-n * Delta ** 2 / 2)))
    mean_err, typ_err = np.array(mean_err), np.array(typ_err)
    s_mean = np.polyfit(ns[30:], np.log(mean_err[30:] * np.sqrt(ns[30:])), 1)[0]      # prefactor ~ n^(-1/2)
    s_typ = np.polyfit(ns[10:30], np.log(typ_err[10:30]), 1)[0]
    print(f"  Gaussian personas, Delta=1: mean error decays like n^(-1/2) exp(-{-s_mean:.4f} n) (Bhattacharyya Delta^2/8 = 0.125), "
          f"typical error like exp(-{-s_typ:.4f} n) (KL = Delta^2/2 = 0.5)")
    assert abs(-s_mean - 0.125) < 1.5e-3 and abs(-s_typ - 0.5) < 1e-6
    n40 = int(np.where(ns == 40)[0][0])
    print(f"  n=40: mean error {mean_err[n40]:.3e}, typical (median) error {typ_err[n40]:.3e}")
    write_table("L7_avg_typ.dat", ["n", "mean", "typical"], [[int(n), a, b] for n, a, b in zip(ns, mean_err, typ_err) if b > 1e-13])
    return dict(mean40=mean_err[n40], typ40=typ_err[n40])


# =====================================================================================================
# E. an atom inside a continuum: power law n^{-d/2}
# =====================================================================================================
def block_E():
    banner("E. a bad atom inside a continuum of benign personas: P(benign | c_n) ~ n^(-d/2)")
    wb, s = 0.05, 1.0
    for d in (1, 2, 3):
        # exact Gaussian integral: M_ben(n) = (1-wb) (1 + n s^2)^(-d/2)
        n = 50.0
        f = lambda r: (1 - wb) * (2 * np.pi * s * s) ** (-d / 2) * np.exp(-r * r / (2 * s * s) - n * r * r / 2) \
            * (2 * np.pi ** (d / 2) / np.exp(gammaln(d / 2))) * r ** (d - 1)
        num = quad(f, 0, 20)[0]
        assert abs(num - (1 - wb) * (1 + n * s * s) ** (-d / 2)) < 1e-12
        ns = np.logspace(8, 12, 9)
        pben = (1 - wb) * (1 + ns * s * s) ** (-d / 2) / (wb + (1 - wb) * (1 + ns * s * s) ** (-d / 2))
        sl = np.polyfit(np.log(ns), np.log(pben), 1)[0]
        print(f"  d={d}: log-log slope of P(benign|c_n) = {sl:.4f}   (prediction -d/2 = {-d / 2})")
        assert abs(sl + d / 2) < 5e-3


# =====================================================================================================
# F. numbers for the exercises and for the boxes
# =====================================================================================================
def block_F():
    banner("F. numbers used in the exercises and boxes")
    out = {}
    # ---- Ex 1: two personas, p_A(1)=0.3, p_B(1)=0.7, prior P(B)=0.02, Delta V = 1, context 6 ones and 2 zeros, beta = 0 and 6
    pA, pB = 0.3, 0.7
    ell = np.log(pB / pA)
    D = (pB - pA) * ell
    L0 = np.log(0.02 / 0.98)
    S = 6 * ell - 2 * ell
    P0 = expit(L0 + S)
    P6 = expit(L0 + S - 6.0)
    print(f"  Ex1: ell = {ell:.5f}, D = {D:.5f}, prior log-odds {L0:.4f}, S = {S:.4f}; P(B|c) = {P0:.4f} (beta=0), {P6:.5f} (beta=6)")
    out["ex1"] = (ell, D, L0, S, P0, P6)
    # exact crossing times (lattice walk with up-prob 0.7, steps +-ell)
    for beta in (0.0, 6.0):
        h = beta * 1.0 - L0
        m = int(np.floor(h / ell)) + 1
        ET = m / (pB - pA)
        print(f"       beta={beta}: h = {h:.4f}, m = {m}, E[T] = {ET:.3f}, Wald bounds [{h / D:.3f}, {(h + ell) / D:.3f}]")
        assert h / D <= ET <= (h + ell) / D
        out[f"ex1_T{int(beta)}"] = (h, m, ET, h / D, (h + ell) / D)
    print(f"       delay of E[T]: {out['ex1_T6'][2] - out['ex1_T0'][2]:.2f};  linear prediction beta Delta V / D = {6.0 / D:.2f};  band +- M/D = {ell / D:.2f}")
    assert abs((out['ex1_T6'][2] - out['ex1_T0'][2]) - 6.0 / D) <= ell / D

    # ---- Ex 2/3: three personas
    pi = np.array([0.5, 0.3, 0.2])
    U = np.array([0.0, 0.5, 1.0])
    V = np.array([0.0, 2.0, 1.0])
    cov = lambda x, y, q: q @ (x * y) - (q @ x) * (q @ y)
    EU, EV = pi @ U, pi @ V
    C = cov(U, V, pi)
    sdU, sdV = np.sqrt(cov(U, U, pi)), np.sqrt(cov(V, V, pi))
    corr = C / (sdU * sdV)

    def tilt(b):
        w = pi * np.exp(-b * V)
        return w / w.sum()

    h = 1e-5
    d1 = (tilt(h) @ U - tilt(-h) @ U) / (2 * h)
    assert abs(d1 + C) < 1e-8
    b = 0.1
    q = tilt(b)
    KL = np.sum(q * np.log(q / pi))
    dP = q @ U - EU
    approx = -np.sqrt(2 * KL) * sdU * corr
    print(f"  Ex2: E U = {EU:.4f}, E V = {EV:.4f}, Cov(U,V) = {C:.4f}; dP/dbeta at 0 = {-C:.4f}; sd U = {sdU:.4f}, sd V = {sdV:.4f}, corr = {corr:.4f}")
    print(f"  Ex3: beta = 0.1: exact dP = {dP:.5f}, KL = {KL:.5f}, -sqrt(2KL) sdU corr = {approx:.5f}")
    out["ex2"] = (EU, EV, C, sdU, sdV, corr, dP, KL, approx)
    # P at beta = 1 and the second derivative
    q1 = tilt(1.0)
    print(f"       beta=1: tilted posterior {np.round(q1, 4)}, P = {q1 @ U:.5f}")
    k3 = np.sum(pi * (U - EU) * (V - EV) ** 2)
    print(f"       kappa(U,V,V) at beta=0: {k3:.5f}")
    out["ex2b"] = (q1, q1 @ U, k3)

    # Ex 3 (continued): the ideal potential V = c U for the same KL budget
    bU = 0.1
    # find beta' such that KL(pi_beta' || pi) equals the KL of the potential V at beta = 0.1, with V' = U
    from scipy.optimize import brentq
    def klU(bb):
        w = pi * np.exp(-bb * U); w /= w.sum()
        return np.sum(w * np.log(w / pi))
    bstar = brentq(lambda bb: klU(bb) - KL, 1e-6, 5.0)
    wU = pi * np.exp(-bstar * U); wU /= wU.sum()
    print(f"       ideal potential V = U with the same KL ({KL:.5f}): beta = {bstar:.4f}, P = {wU @ U:.5f}, drop {wU @ U - EU:.5f} (first-order estimate {-np.sqrt(2 * KL) * sdU:.5f})")
    out["ex3b"] = (bstar, wU @ U - EU)

    # Ex 6: lattice walk, beta = 8  (prior P(B)=0.1, Delta V = 1, p_A(1)=1/3, p_B(1)=2/3)
    ln2 = np.log(2.0)
    Dlat = ln2 / 3
    L0lat = np.log(0.1 / 0.9)
    h8 = 8.0 - L0lat
    m8 = int(np.floor(h8 / ln2)) + 1
    print(f"  Ex6: beta=8: h = {h8:.4f}, m = {m8}, E T = {3 * m8}, Var T = {24 * m8} (sd {np.sqrt(24 * m8):.2f}); Wald bounds [{h8 / Dlat:.3f}, {(h8 + ln2) / Dlat:.3f}]")
    assert 3 * m8 > h8 / Dlat and 3 * m8 <= (h8 + ln2) / Dlat
    out["ex6"] = (h8, m8, 3 * m8, 24 * m8, h8 / Dlat, (h8 + ln2) / Dlat)

    # ---- Ex 4: Gibbs principle numbers
    pi4 = np.array([0.5, 0.3, 0.2])
    V4 = np.array([2.0, 1.0, 0.0])
    Z = np.sum(pi4 * np.exp(-V4))
    q4 = pi4 * np.exp(-V4) / Z
    print(f"  Ex4: Z = {Z:.5f}, tilted = {np.round(q4, 4)}, min of beta E[V] + KL = -log Z = {-np.log(Z):.5f}; "
          f"value at q = pi: {pi4 @ V4:.4f}; at the V-minimizer: {np.log(1 / pi4[2]):.4f}")
    out["ex4"] = (Z, q4, -np.log(Z))

    # ---- Ex 5: exact Gaussian closed form  n dE[U] = -beta a v n/(n + s^-2)
    s, a_, v_, beta5 = 2.0, 1.0, 1.0, 3.0
    for n in (10, 25, 100):
        sig2 = 1.0 / (n + s ** -2)
        # exact mean shift of the tilted Gaussian posterior (U = a z + const, V = v z)
        dE = -beta5 * a_ * v_ * sig2
        assert abs(n * dE - (-beta5 * a_ * v_ * n / (n + s ** -2))) < 1e-15
        print(f"  Ex5: n={n}: n*dE = {n * dE:.4f} (limit {-beta5 * a_ * v_}), relative gap {abs(n * dE / (-beta5 * a_ * v_) - 1):.4f}")
    n_min = int(np.ceil(99 * s ** -2))
    assert abs(n_min / (n_min + s ** -2) - 0.99) < 0.01 and (n_min - 1) / (n_min - 1 + s ** -2) < 0.99 <= n_min / (n_min + s ** -2)
    print(f"       smallest n with relative gap <= 1%: {n_min}")
    # verify the Gaussian closed form against a grid
    grid = np.linspace(-6, 6, 400001)
    for n in (10, 25):
        sig2 = 1.0 / (n + s ** -2)
        xbar = 0.7
        mu = sig2 * n * xbar
        base = -(grid - mu) ** 2 / (2 * sig2)
        res = []
        for bb in (0.0, beta5):
            lw = base - bb * v_ * grid
            wt = np.exp(lw - lw.max())
            wt /= wt.sum()
            res.append(wt @ (a_ * grid))
        assert abs((res[1] - res[0]) - (-beta5 * a_ * v_ * sig2)) < 1e-8
    out["ex5"] = n_min

    # ---- Ex 7: SFT as a tilt, V_D(z) = -log p(D|z):  plain vs educational insecure-code data
    pi7 = np.array([0.97, 0.03])                   # benign H, malicious M
    U7 = np.array([0.02, 0.80])
    plain = np.log(np.array([0.05, 0.5]))          # log p(insecure completion | z, plain request)
    edu = np.log(np.array([0.5, 0.5]))             # educational request: the benign persona complies as readily
    res7 = {}
    for name, ll in (("plain", plain), ("educational", edu)):
        row = []
        for b in range(0, 6):
            w = pi7 * np.exp(b * ll)
            w /= w.sum()
            row.append(w @ U7)
        res7[name] = row
        C0 = np.cov(np.vstack([U7, ll]), aweights=pi7, bias=True)[0, 1]
        print(f"  Ex7 [{name}]: P(bad) for beta = 0..5: {np.round(row, 4)};  dP/dbeta at 0 = Cov(U, log p(D|z)) = {C0:.5f}")
    out["ex7"] = res7

    # ---- Ex 8: inoculation as a conditional tilt (honest H, obedient O, cheater C)
    pi8 = np.array([0.90, 0.08, 0.02])
    U8 = np.array([0.01, 0.02, 0.60])
    hack = {"plain": np.array([0.02, 0.05, 0.95]), "inoculated": np.array([0.50, 0.95, 0.95])}
    res8 = {}
    for name, ph in hack.items():
        row = []
        for b in (0, 1, 2, 3):
            w = pi8 * ph ** b
            w /= w.sum()
            row.append(float(w @ U8))
        C0 = float(pi8 @ (U8 * np.log(ph)) - (pi8 @ U8) * (pi8 @ np.log(ph)))
        res8[name] = (row, C0)
        print(f"  Ex8 [{name}]: P(bad) for beta = 0..3: {np.round(row, 4)};  slope at 0 = Cov(U, log p(hack|z,c)) = {C0:.4f}")
    out["ex8"] = res8

    # Ex 8(c): a 'do not hack' instruction lowers the hack probability of the honest and obedient personas
    ph = np.array([0.005, 0.01, 0.95])
    w = pi8 * ph ** 2
    w /= w.sum()
    print(f"  Ex8(c): do-not-hack instruction (0.005, 0.01, 0.95): P_2 = {w @ U8:.4f}  (plain {res8['plain'][0][2]:.4f}); LR C:H = {0.95 / 0.005:.0f}, C:O = {0.95 / 0.01:.0f}")
    out["ex8c"] = float(w @ U8)
    assert w @ U8 > res8['plain'][0][2]

    # Ex 10(b): atom inside a continuum, d = 2, n = 1000
    wb_, d_, n_ = 0.05, 2, 1000
    Mb = (1 - wb_) * (1 + n_) ** (-d_ / 2)
    print(f"  Ex10(b): P(benign | c_n) at d=2, n=1000: {Mb / (wb_ + Mb):.5f}")
    out["ex10b"] = Mb / (wb_ + Mb)

    # ---- Ex 9: average versus typical for Gaussian personas at distance 1 (see block_C2)
    # ---- Ex 11: Waluigi-type prior-odds bound
    c_bits, Dtok, bdv = 20.0, 0.05, 30.0
    n_no = c_bits * np.log(2) / Dtok
    n_tr = (c_bits * np.log(2) + bdv) / Dtok
    print(f"  Ex11: tokens to overturn the prior (no training) = {n_no:.1f}; with beta Delta V = 30 nats: {n_tr:.1f}")
    out["ex11"] = (n_no, n_tr)

    # ---- Ex 12: mixture of exponentials: int_0^1 D^(alpha-1) e^{-nD} dD ~ Gamma(alpha) n^{-alpha}
    for alpha in (0.5, 1.0, 1.5):
        n = 1000.0
        val = quad(lambda x: x ** (alpha - 1) * np.exp(-n * x), 0, 1, limit=200)[0]
        ref = np.exp(gammaln(alpha)) * n ** (-alpha)
        print(f"  Ex12: alpha={alpha}: integral {val:.6e} vs Gamma(alpha) n^-alpha = {ref:.6e}")
        assert abs(val / ref - 1) < 1e-6
    return out


# =====================================================================================================
# G. a circle of personas: the first-order effect averaged over the context persona is a Dirichlet form
# =====================================================================================================
def block_G():
    banner("G. circle of personas: n * dP -> -beta sigma^2 U'(theta*) V'(theta*); average = Dirichlet form")
    sigma, beta, n = 1.0, 1.5, 4000
    U = lambda t: 0.5 + 0.3 * np.cos(t) + 0.1 * np.sin(2 * t)
    V = lambda t: np.cos(t) + 0.5 * np.cos(2 * t)
    dU = lambda t: -0.3 * np.sin(t) + 0.2 * np.cos(2 * t)
    dV = lambda t: -np.sin(t) - np.sin(2 * t)
    th = np.linspace(0, 2 * np.pi, 40001)[:-1]
    errs, vals = [], []
    for ts in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        ll = -(n / sigma ** 2) * (1 - np.cos(th - ts))             # typical context of persona theta*, uniform prior
        res = []
        for b in (0.0, beta):
            lw = ll - b * V(th)
            wt = np.exp(lw - lw.max())
            wt /= wt.sum()
            res.append(wt @ U(th))
        pred = -beta * sigma ** 2 * dU(ts) * dV(ts)
        vals.append(n * (res[1] - res[0]))
        errs.append(abs(n * (res[1] - res[0]) - pred))
    avg = np.mean(vals)
    pred_avg = -beta * sigma ** 2 * 0.15
    print(f"  max error of n dP against -beta sigma^2 U' V' over 48 context personas: {max(errs):.2e}; mean over theta*: {avg:.5f}; Dirichlet prediction {pred_avg:.5f}")
    assert max(errs) < 2e-3 and abs(avg - pred_avg) < 1e-3
    # Fourier form of the Dirichlet pairing: sum_k k^2 (cos/sin coefficient products) / 2
    assert abs(0.5 * 1 ** 2 * 0.3 * 1.0 - 0.15) < 1e-15


def main():
    A = block_A()
    B = block_B()
    C = block_C()
    block_E()
    block_G()
    F = block_F()
    banner("all assertions passed; data files written to figures/data/L7_*.dat")


if __name__ == "__main__":
    main()
