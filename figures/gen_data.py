#!/usr/bin/env python3
"""Generate the numerical data behind the figures in the lecture notes.

Run from the project root:   python3 figures/gen_data.py
Needs only numpy and scipy.  Output goes to figures/data/*.dat (whitespace-separated tables
that pgfplots reads directly).

Groups of data
--------------
1. Spectral model of Lecture 2:     lambda_k = k^-alpha,  a_k^2 = k^(-1-alpha*beta),
   loss L(t) = 1/2 * sum_k a_k^2 exp(-2 lambda_k t)   (exact sum + closed-form asymptote).
2. Bayes regret of Lecture 3:       finite hypothesis class vs. one-parameter Bernoulli model.
3. Compute-optimal frontier of Lecture 2 (illustration with the published Chinchilla parametric fit).
4. Lecture 4: closed-loop Jones growth model (trajectories, blow-up time; closed form checked against
   numerical integration of the original ODE) and the CES production function (curves, isoquants).
5. Lecture 5: the Gaussian example (exact Bayes regret checked by Monte Carlo), Langevin sampling of its posterior,
   and the Fourier ("Dirichlet kernel") picture behind modular addition.
   (The grokking experiment of Lecture 5 is a separate script, figures/grokking.py, because it needs PyTorch.)
"""
import math
import os

import numpy as np
from scipy.special import gammaln, gammainc, logsumexp
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)


def save(name, header, cols):
    arr = np.column_stack(cols)
    path = os.path.join(OUT, name)
    np.savetxt(path, arr, header=header, comments="", fmt="%.8e")
    print(f"wrote {path}  shape={arr.shape}")


# ----------------------------------------------------------------------------------------------
# 1. spectral model
# ----------------------------------------------------------------------------------------------
ALPHA, BETA = 1.5, 0.5          # lambda_k ~ k^-alpha ; a_k^2 ~ k^(-1 - alpha*beta)
K_MAX = 4_000_000               # explicit modes; analytic tail added below


def spectral_loss(t):
    """0.5 * sum_{k>=1} k^(-1-alpha*beta) * exp(-2 t k^-alpha)  (exact up to the analytic tail)."""
    total = 0.0
    chunk = 1_000_000
    for start in range(1, K_MAX + 1, chunk):
        k = np.arange(start, min(start + chunk, K_MAX + 1), dtype=np.float64)
        total += np.sum(k ** (-1.0 - ALPHA * BETA) * np.exp(-2.0 * t * k ** (-ALPHA)))
    # tail: sum_{k>K_MAX} ~ integral_{K_MAX+1/2}^inf x^(-1-alpha beta) exp(-2 t x^-alpha) dx
    #   = (2t)^(-beta)/alpha * Gamma(beta) * P(beta, 2t (K_MAX+1/2)^-alpha)      (substitute u = 2t x^-alpha;
    #   P = regularized lower incomplete gamma function)
    u0 = 2.0 * t * (K_MAX + 0.5) ** (-ALPHA)
    tail = (2.0 * t) ** (-BETA) / ALPHA * math.gamma(BETA) * gammainc(BETA, u0)
    return 0.5 * (total + tail)


def spectral_asymptote(t):
    # 1/2 * (1/alpha) * Gamma(beta) * 2^(-beta) * t^(-beta)
    return math.gamma(BETA) / (ALPHA * 2.0 ** (BETA + 1.0)) * t ** (-BETA)


# (a) eigenvalues and target coefficients
k = np.unique(np.round(np.logspace(0, 4, 120)).astype(int)).astype(float)
save("spec_modes.dat", "k lam a2", [k, k ** (-ALPHA), k ** (-1.0 - ALPHA * BETA)])

# (b) learned fraction 1 - exp(-lambda_k t) for several t
kk = np.logspace(0, 4.2, 200)
ts = [1e1, 1e2, 1e3, 1e4]
save("spec_learned.dat", "k " + " ".join(f"t{int(math.log10(t))}" for t in ts),
     [kk] + [1.0 - np.exp(-t * kk ** (-ALPHA)) for t in ts])

# (c) loss vs t
tt = np.logspace(0, 6, 49)
exact = np.array([spectral_loss(t) for t in tt])
asym = np.array([spectral_asymptote(t) for t in tt])
save("spec_loss.dat", "t Lexact Lasym ratio", [tt, exact, asym, exact / asym])
print("   spectral check (alpha=%.2f beta=%.2f):  L*t^beta at t=1e6 -> %.6f ; predicted constant %.6f"
      % (ALPHA, BETA, exact[-1] * tt[-1] ** BETA, math.gamma(BETA) / (ALPHA * 2 ** (BETA + 1))))
loc_slope = np.diff(np.log(exact)) / np.diff(np.log(tt))
print("   local log-log slope at the end: %.4f  (expected -beta = %.2f)" % (loc_slope[-1], -BETA))

# ----------------------------------------------------------------------------------------------
# 2. Bayes regret (all quantities in bits)
# ----------------------------------------------------------------------------------------------
LN2 = math.log(2.0)
THETA = 5.0 / 17.0                       # true Bernoulli parameter
GRID = np.arange(1, 17) / 17.0           # finite hypothesis class {j/17}, uniform prior 1/16
J_STAR = 4                               # index of 5/17 in GRID


def regret_finite(n):
    ks = np.arange(0, n + 1)
    logpmf = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1) \
        + ks * math.log(THETA) + (n - ks) * math.log(1 - THETA)
    # log of the mixture likelihood of one sequence with k ones (per-sequence, not per-type)
    ll = np.array([ks * math.log(g) + (n - ks) * math.log(1 - g) for g in GRID])   # (16, n+1)
    log_mix = logsumexp(ll, axis=0) - math.log(len(GRID))
    log_true = ks * math.log(THETA) + (n - ks) * math.log(1 - THETA)
    return float(np.sum(np.exp(logpmf) * (log_true - log_mix)) / LN2)


def regret_uniform_prior(n):
    ks = np.arange(0, n + 1)
    logpmf = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1) \
        + ks * math.log(THETA) + (n - ks) * math.log(1 - THETA)
    log_true = ks * math.log(THETA) + (n - ks) * math.log(1 - THETA)
    log_mix = -(math.log(n + 1) + gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1))   # = log Z_n
    return float(np.sum(np.exp(logpmf) * (log_true - log_mix)) / LN2)


def regret_asymptotic(n):
    # Clarke-Barron:  d/2 log(n / (2 pi e)) + log( sqrt(det I) / w(theta) ),  d = 1, w = 1 (uniform prior)
    fisher = 1.0 / (THETA * (1.0 - THETA))
    return (0.5 * math.log(n / (2 * math.pi * math.e)) + 0.5 * math.log(fisher)) / LN2


ns = np.unique(np.round(np.logspace(0, 5, 41)).astype(int))
Rf = np.array([regret_finite(int(n)) for n in ns])
Rc = np.array([regret_uniform_prior(int(n)) for n in ns])
Ra = np.array([regret_asymptotic(int(n)) for n in ns])
save("regret.dat", "n Rfinite Rcont Rasym", [ns.astype(float), Rf, Rc, Ra])
print("   finite class: R_n(1e5) = %.4f bits  (bound log2(16) = 4)" % Rf[-1])
print("   uniform prior: R_n(1e5) = %.4f bits ; asymptotic formula = %.4f bits ; gap = %.2e"
      % (Rc[-1], Ra[-1], Rc[-1] - Ra[-1]))

# ----------------------------------------------------------------------------------------------
# 3. compute-optimal frontier (Chinchilla parametric fit, used purely as an illustration)
# ----------------------------------------------------------------------------------------------
E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28


def loss_ND(N, D):
    return E + A / N ** a + B / D ** b


Ns = [1e7, 1e8, 1e9, 1e10, 1e11]
cols = []
Cgrid = np.logspace(15, 26, 111)
for N in Ns:
    D = Cgrid / (6.0 * N)
    L = loss_ND(N, D)
    L[(D < 5e7) | (D > 3e13)] = np.nan           # plot each N only over a plausible data range
    cols.append(L)
# compute-optimal envelope (closed form)
Nstar = (a * A / (b * B * 6.0 ** b)) ** (1.0 / (a + b)) * Cgrid ** (b / (a + b))
Dstar = Cgrid / (6.0 * Nstar)
# Everything is saved as the *reducible* loss L - E: only L - E is a pure power law in C along the frontier
# (L* - E = kappa * C^(-a b / (a + b)) exactly), whereas L* itself bends towards the irreducible loss E.
cols_red = [c - E for c in cols]
save("chinchilla_curves.dat", "C " + " ".join(f"N{int(math.log10(N))}" for N in Ns) + " Lstar Nstar Dstar",
     [Cgrid] + cols_red + [loss_ND(Nstar, Dstar) - E, Nstar, Dstar])
slope = np.diff(np.log(loss_ND(Nstar, Dstar) - E)) / np.diff(np.log(Cgrid))
print("   frontier: d log(L*-E) / d log C = %.5f (constant); a*b/(a+b) = %.5f" % (slope.mean(), a * b / (a + b)))
print("   frontier: local slope of the TOTAL loss L* ranges from %.4f to %.4f"
      % ((np.diff(np.log(loss_ND(Nstar, Dstar))) / np.diff(np.log(Cgrid)))[np.argmin(abs(Cgrid - 1e18))],
         (np.diff(np.log(loss_ND(Nstar, Dstar))) / np.diff(np.log(Cgrid)))[-1]))
print("   frontier exponents: N* ~ C^%.3f, D* ~ C^%.3f ; tokens per parameter D*/N* at C=1e23: %.1f"
      % (b / (a + b), a / (a + b), (Dstar / Nstar)[np.argmin(abs(Cgrid - 1e23))]))
print("   sanity: 70B params / 1.4T tokens -> L = %.3f" % loss_ND(70e9, 1.4e12))


# ----------------------------------------------------------------------------------------------
# 4. Lecture 4: Jones growth model with the loop closed (R = C S) and the CES production function
# ----------------------------------------------------------------------------------------------
# Model:  dS/dt = S * R^lam * S^(-beta).  For ANY research input R(t):  d(S^beta)/dt = beta * R^lam.
# With R = C*S the variable u = S^beta solves  du/dt = k u^r,  k = beta C^lam,  r = lam/beta.
def jones_u(t, r, k=1.0, u0=1.0):
    """Closed form of du/dt = k u^r (NaN after blow-up)."""
    t = np.asarray(t, dtype=float)
    if abs(r - 1.0) < 1e-12:
        return u0 * np.exp(k * t)
    base = u0 ** (1.0 - r) + (1.0 - r) * k * t
    out = np.full_like(t, np.nan)
    ok = base > 0
    out[ok] = base[ok] ** (1.0 / (1.0 - r))
    return out


# (a) check the closed form against direct integration of the ORIGINAL equation dS/dt = S R^lam S^-beta, R = C S
worst = 0.0
for lam, beta, C, S0 in [(0.8, 0.5, 2.0, 1.0), (1.0, 0.7, 0.5, 3.0), (0.4, 0.8, 1.5, 2.0), (0.9, 0.9, 1.0, 1.0)]:
    r, k, u0 = lam / beta, beta * C ** lam, S0 ** beta
    Tstar = (u0 ** (1 - r) / ((r - 1) * k)) if r > 1 else np.inf
    t_end = 0.9 * Tstar if np.isfinite(Tstar) else 20.0
    sol = solve_ivp(lambda t, y: y * (C * y) ** lam * y ** (-beta), [0, t_end], [S0], rtol=1e-12, atol=1e-14,
                    method="LSODA", t_eval=np.linspace(0, t_end, 7))
    u_closed = jones_u(sol.t, r, k, u0)
    worst = max(worst, float(np.max(np.abs(sol.y[0] ** beta / u_closed - 1.0))))
print("   Jones loop: closed form vs ODE integration, max relative error %.2e" % worst)
assert worst < 1e-8

# (b) trajectories u(t) for several returns r   (units: k = 1, u0 = 1; blow-up at T* = 1/(r-1))
tt = np.linspace(0.0, 3.0, 301)
rs = [0.5, 1.0, 1.5, 2.0, 3.0]
cols = []
for r in rs:
    u = jones_u(tt, r)
    u[u > 1e4] = np.nan                       # clip for plotting
    cols.append(u)
save("jones_traj.dat", "t " + " ".join("r%s" % str(r).replace(".", "p") for r in rs), [tt] + cols)
print("   blow-up times T* = 1/(r-1):", {r: 1.0 / (r - 1.0) for r in rs if r > 1})


# (c) CES production function  R = (a K^rho + (1-a) L^rho)^(1/rho)
def ces(K, L, rho, a=0.5):
    K = np.asarray(K, float); L = np.asarray(L, float)
    if abs(rho) < 1e-12:
        return K ** a * L ** (1.0 - a)
    return (a * K ** rho + (1.0 - a) * L ** rho) ** (1.0 / rho)


# limits (checked numerically): rho -> 0 Cobb-Douglas, rho -> -inf min, rho = 1 linear; ceiling a^(1/rho) K for rho < 0
Kt, Lt, at = 2.0, 7.0, 0.3
assert abs(ces(Kt, Lt, 1e-8, at) - Kt ** at * Lt ** (1 - at)) < 1e-6
assert abs(ces(Kt, Lt, -400.0, at) - min(Kt, Lt)) < 0.02
assert abs(ces(Kt, Lt, 1.0, at) - (at * Kt + (1 - at) * Lt)) < 1e-12
for rho in (-0.4, -1.0, -4.0):
    assert abs(ces(Kt, 1e40, rho, at) / (at ** (1 / rho) * Kt) - 1.0) < 1e-6
print("   CES limits verified; equal weights, rho=-0.4: ceiling/R(K,K) = %.4f (= 2^2.5)" % 2 ** 2.5)

rhos = [1.0, 0.5, 0.0, -0.4, -1.0, -4.0]
Lg = np.logspace(-1, 3, 161)
save("ces_labor.dat", "L " + " ".join("rho%s" % ("m%g" % -r if r < 0 else "%g" % r).replace(".", "p") for r in rhos),
     [Lg] + [ces(1.0, Lg, r) for r in rhos])

# isoquants R = 1 (equal weights), L as a function of K
Kg = np.linspace(0.02, 2.4, 240)
iso = {}
iso["rho1"] = 2.0 - Kg                                              # (K + L)/2 = 1
iso["rho0"] = 1.0 / Kg                                              # sqrt(K L) = 1
with np.errstate(divide="ignore", invalid="ignore"):
    iso["rhom1"] = np.where(Kg > 0.5, Kg / (2 * Kg - 1), np.nan)    # 1/K + 1/L = 2
    inner = 2.0 - Kg ** (-4.0)
    iso["rhom4"] = np.where(inner > 0, inner ** (-0.25), np.nan)    # K^-4 + L^-4 = 2
iso["rho1"] = np.where(iso["rho1"] > 0, iso["rho1"], np.nan)
for name, rho in [("rho1", 1.0), ("rho0", 0.0), ("rhom1", -1.0), ("rhom4", -4.0)]:
    ok = ~np.isnan(iso[name])
    assert np.max(np.abs(ces(Kg[ok], iso[name][ok], rho) - 1.0)) < 1e-9, name
save("ces_isoquant.dat", "K " + " ".join(iso.keys()), [Kg] + list(iso.values()))


# ----------------------------------------------------------------------------------------------
# 5. Lecture 5
# ----------------------------------------------------------------------------------------------
# (a) Gaussian example: q = N(theta*, I_d), model N(theta, I_d), prior N(0, I_d).  Exact Bayes regret (nats):
#        R_n = d/2 log(n+1) + n/(2(n+1)) (|theta*|^2 - d)
rng5 = np.random.default_rng(20260928)
for d_, n_, ts_ in [(1, 5, 1.3), (2, 20, 0.7), (3, 100, 2.0)]:
    ts_v = ts_ * np.ones(d_)
    M = 200000
    X = ts_v + rng5.standard_normal((M, n_, d_))
    xbar = X.mean(axis=1)
    logq = -0.5 * n_ * d_ * math.log(2 * math.pi) - 0.5 * np.sum((X - ts_v) ** 2, axis=(1, 2))
    logpB = (-0.5 * n_ * d_ * math.log(2 * math.pi) - 0.5 * d_ * math.log(n_ + 1) - 0.5 * np.sum(X ** 2, axis=(1, 2))
             + n_ ** 2 / (2 * (n_ + 1)) * np.sum(xbar ** 2, axis=1))
    mc = (logq - logpB)
    formula = 0.5 * d_ * math.log(n_ + 1) + n_ / (2 * (n_ + 1)) * (np.sum(ts_v ** 2) - d_)
    assert abs(mc.mean() - formula) < 5 * mc.std() / math.sqrt(M)
print("   Gaussian example: exact Bayes regret formula agrees with Monte Carlo (5 standard errors)")

# (b) Langevin dynamics for the posterior of the one-dimensional Gaussian example
n_obs = 20
x_obs = 1.0 + rng5.standard_normal(n_obs)
a_post = n_obs + 1.0                                   # posterior precision
mu_post = n_obs * x_obs.mean() / (n_obs + 1.0)
dU = lambda th: -np.sum(x_obs - th) + th               # U = 1/2 sum (x_i - th)^2 + 1/2 th^2
eps, T, burn = 0.002, 3_000_000, 20_000
noise = rng5.standard_normal(T)
th, trace = 0.0, np.empty(T)
for k in range(T):
    th = th - eps * dU(th) + math.sqrt(2 * eps) * noise[k]
    trace[k] = th
tr_ = trace[burn:]
var_ula = 1.0 / (a_post * (1.0 - eps * a_post / 2.0))  # stationary variance of the Euler-Maruyama chain
print("   Langevin: exact posterior N(%.4f, %.5f); samples mean %.4f var %.5f; predicted ULA variance %.5f"
      % (mu_post, 1 / a_post, tr_.mean(), tr_.var(), var_ula))
assert abs(tr_.mean() - mu_post) < 0.01 and abs(tr_.var() / var_ula - 1) < 0.03
sd = 1.0 / math.sqrt(a_post)
edges = np.linspace(mu_post - 4 * sd, mu_post + 4 * sd, 41)
cnt, _ = np.histogram(tr_, bins=edges, density=True)
ctr = 0.5 * (edges[1:] + edges[:-1])
exact = np.exp(-0.5 * a_post * (ctr - mu_post) ** 2) * math.sqrt(a_post / (2 * math.pi))
save("langevin_hist.dat", "theta empirical exact", [ctr, cnt, exact])
gd = 0.0
for _ in range(5000):
    gd = gd - 0.05 * dU(gd)
print("   noise-free gradient descent ends at %.5f = MAP = posterior mean %.5f" % (gd, mu_post))
with open(os.path.join(OUT, "langevin_info.dat"), "w") as fh:        # numbers quoted in the caption
    fh.write("mu_post %.6f\nsd_post %.6f\nmap %.6f\n" % (mu_post, sd, gd))

# (c) Fourier partial sums  S_K(t) = (2/p) sum_{k=1}^K cos(2 pi k t / p),  p = 23;  K = (p-1)/2 gives [t = 0 mod p] - 1/p exactly
p5 = 23
tg = np.linspace(-11.5, 11.5, 461)
Ks = [1, 2, 4, 11]
parts = [(2.0 / p5) * sum(np.cos(2 * np.pi * k * tg / p5) for k in range(1, K + 1)) for K in Ks]
ti = np.arange(-11, 12)
exact_delta = (ti % p5 == 0).astype(float) - 1.0 / p5
full_at_int = (2.0 / p5) * sum(np.cos(2 * np.pi * k * ti / p5) for k in range(1, 12))
assert np.max(np.abs(full_at_int - exact_delta)) < 1e-12
save("fourier_partial.dat", "t " + " ".join("K%d" % K for K in Ks), [tg] + parts)
save("fourier_exact.dat", "t delta", [ti.astype(float), exact_delta])
print("   Fourier identity (2/p) sum_{k<=(p-1)/2} cos(2 pi k t/p) = [t=0 mod p] - 1/p verified for p = 23")
