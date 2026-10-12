#!/usr/bin/env python3
"""Numerical checks for the 'What a bottleneck does to the explosion condition' box of Lecture 4.

Run from the project root:   python3 figures/check_L4_bottleneck.py      (numpy + scipy; ~20 s)

Model: u = S^beta,  du/dt = beta * R^lambda,  R = CES(K, L) with L = C*S = C*u^(1/beta),
       CES(K, L) = (alpha K^rho + (1-alpha) L^rho)^(1/rho)   (rho = 0: K^alpha L^(1-alpha)).
Claims checked
  (A) K constant: finite-time blow-up  <=>  theta*r > 1, theta = 1 (0<rho<=1), 1-alpha (rho=0), 0 (rho<0).
  (B) rho < 0, K = K0 exp(g_K t), g_K > 0:  (1/t) log S -> r*g_K if r > 1;  -> 0 if r < 1.
  (C) the bound is not uniform in rho: ceiling 2^(1/|rho|) K and the blow-up time for rho -> 0+.
"""
import math

import numpy as np
from scipy.integrate import quad, solve_ivp
import warnings
warnings.filterwarnings('ignore')


def ces(K, L, rho, a=0.5):
    if rho == 0.0:
        return K ** a * L ** (1 - a)
    return (a * K ** rho + (1 - a) * L ** rho) ** (1.0 / rho)


def log_ces(logK, logL, rho, a=0.5):
    """log of CES computed from logs (stable for huge values)."""
    if rho == 0.0:
        return a * logK + (1 - a) * logL
    z = np.array([math.log(a) + rho * logK, math.log(1 - a) + rho * logL])
    m = z.max()
    return (m + math.log(np.exp(z - m).sum())) / rho


# ---------------------------------------------------------------------------------------------------
# (A) constant K: the equation du/dt = beta R^lambda = f(u) is autonomous, so (Osgood) it blows up in finite time
#     iff  int^infty du / f(u) < infinity.  With x = log u the integrand is g(x) = e^x / (beta R^lambda); it decays
#     exponentially iff theta*r > 1 (and tends to a constant or grows otherwise), so we look at its slope at large x.
# ---------------------------------------------------------------------------------------------------
def log_integrand(x, rho, lam, beta=1.0, K=1.0, C=1.0, a=0.5):
    logL = math.log(C) + x / beta
    return x - math.log(beta) - lam * log_ces(math.log(K), logL, rho, a)


def blows_up(rho, lam, **kw):
    x1, x2 = 200.0, 400.0
    slope = (log_integrand(x2, rho, lam, **kw) - log_integrand(x1, rho, lam, **kw)) / (x2 - x1)
    return slope < -1e-3, slope


print("(A) blow-up classification, K constant (beta=1, K=C=1, alpha=1/2):  rho x r -> blow-up?  (expected: theta*r>1)")
ok = True
for rho in [1.0, 0.5, 0.0, -0.5, -2.0]:
    theta = 1.0 if rho > 0 else (0.5 if rho == 0 else 0.0)
    row = []
    for r in [0.5, 1.0, 1.5, 1.9, 2.1]:
        got, slope = blows_up(rho, lam=r)       # beta = 1  =>  r = lambda ;  slope = 1 - theta*r  (at large x)
        want = theta * r > 1
        assert abs(slope - (1 - theta * r)) < 1e-6, (rho, r, slope)
        ok &= (got == want)
        row.append(f"{r}:{'B' if got else '-'}")
    print(f"   rho={rho:5.1f}  theta={theta:3.1f}   " + "  ".join(row))
assert ok, "classification disagrees with theta*r>1"
print("   all 25 cases agree with 'blow-up iff theta*r > 1' (the log-slope of the integrand equals 1 - theta*r)")

# ---------------------------------------------------------------------------------------------------
# (B) rho < 0, K = K0 exp(g_K t): late-time growth rate of log S
# ---------------------------------------------------------------------------------------------------
def late_rate(rho, lam, beta=1.0, gK=0.1, C=1.0, a=0.5, t1=300.0, t2=400.0):
    def rhs(t, y):
        x = y[0]                                   # x = log u
        logK = gK * t
        logL = math.log(C) + x / beta
        logR = log_ces(logK, logL, rho, a)
        return [beta * math.exp(lam * logR - x)]
    sol = solve_ivp(rhs, [0, t2], [0.0], t_eval=[t1, t2], rtol=1e-10, atol=1e-12, method="LSODA")
    x1, x2 = sol.y[0]
    return (x2 - x1) / beta / (t2 - t1)             # slope of log S = x/beta


print("(B) rho<0, K=exp(g_K t), g_K=0.1, beta=1: late-time slope of log S")
for rho in [-0.4, -1.0, -3.0]:
    s = late_rate(rho, lam=1.5)                      # r = 1.5 > 1 : expect r*g_K = 0.15
    print(f"   rho={rho:5.1f}  r=1.5   slope={s:.4f}   (r g_K = 0.1500)")
    assert abs(s - 0.15) < 0.003
    s = late_rate(rho, lam=1.2)                      # r = 1.2 : expect 0.12
    print(f"   rho={rho:5.1f}  r=1.2   slope={s:.4f}   (r g_K = 0.1200)")
    assert abs(s - 0.12) < 0.003
    s = late_rate(rho, lam=0.5)                      # r = 0.5 < 1 : expect -> 0 (polynomial)
    print(f"   rho={rho:5.1f}  r=0.5   slope={s:.4f}   (limit 0: labor-limited, polynomial growth)")
    assert s < 0.01

# ---------------------------------------------------------------------------------------------------
# (C) non-uniformity in rho
# ---------------------------------------------------------------------------------------------------
print("(C) ceiling 2^(1/|rho|) K for alpha=1/2:", {r: 2 ** (1 / abs(r)) for r in [-0.4, -0.1, -0.05]})
assert abs(2 ** 2.5 - 5.657) < 1e-3 and 2 ** 10 == 1024 and 2 ** 20 == 1048576


def blowup_time(rho, lam=1.5, beta=1.0, K=1.0, C=1.0, a=0.5, S0=1.0):
    """T* = int_{S0}^infty ds / (beta^{-1} ... ): for beta=1, du/dt = R^lam with u=S, so T* = int ds / R(K,CS)^lam."""
    assert beta == 1.0
    f = lambda s: ces(K, C * s, rho, a) ** (-lam)
    # integrate on a log grid to reach very large s
    val, err = quad(lambda y: f(math.exp(y)) * math.exp(y), math.log(S0), math.log(1e300), limit=500, epsabs=0, epsrel=1e-10)
    return val


print("    blow-up time T* (lambda=1.5, beta=1, alpha=1/2, K=C=S0=1) as rho -> 0+:")
ref = {0.5: 6.0, 0.2: 12.9, 0.1: 29.8, 0.05: 104.0, 0.02: 2152.0}
for rho in [0.5, 0.2, 0.1, 0.05, 0.02]:
    T = blowup_time(rho)
    print(f"   rho={rho:5.2f}   T* = {T:10.2f}")
    assert abs(T / ref[rho] - 1) < 0.03, (rho, T, ref[rho])
print("   (compare Cobb-Douglas rho=0: theta r = (1-alpha) r = 0.75 < 1, no blow-up: T* -> infinity)")
print("OK: all checks passed")
