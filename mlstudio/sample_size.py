"""How many patients do I need to develop a prediction model?

Binary outcome: the three criteria of Riley et al. (BMJ 2020;368:m441), the same as the 'pmsampsize' R/Stata package.
  (i)  expected uniform shrinkage >= 0.90 (little overfitting)
  (ii) small absolute difference (<= 0.05) between apparent and adjusted Nagelkerke R2
  (iii) precise estimate of the overall outcome risk (margin of error <= 0.05)
The anticipated Cox-Snell R2 can be given directly, as a Nagelkerke R2, or approximated from an expected C-statistic
(AUC) by simulation, following Riley, Van Calster & Collins (Stat Med 2021).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def max_r2_cs(prevalence: float) -> float:
    p = prevalence
    ln_null = p * math.log(p) + (1 - p) * math.log(1 - p)
    return 1 - math.exp(2 * ln_null)


def r2cs_from_auc(auc: float, prevalence: float, n: int = 200_000, seed: int = 1) -> float:
    """Approximate Cox-Snell R2 of a well-calibrated model with the given C-statistic."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    d = math.sqrt(2) * norm.ppf(auc)
    lp = rng.normal(0, 1, n) + d * y
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    p = np.clip(m.predict_proba(lp.reshape(-1, 1))[:, 1], 1e-12, 1 - 1e-12)
    ll_full = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    pbar = y.mean()
    ll_null = np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar))
    lr = 2 * (ll_full - ll_null)
    return float(1 - math.exp(-lr / n))


def riley_binary(prevalence: float, parameters: int, r2cs: float, shrinkage: float = 0.9, delta: float = 0.05,
                 margin: float = 0.05) -> dict:
    p, phi = parameters, prevalence
    mx = max_r2_cs(phi)
    if not 0 < r2cs < mx:
        raise ValueError(f"The anticipated Cox-Snell R2 must be between 0 and {mx:.3f} for this prevalence.")
    n1 = p / ((shrinkage - 1) * math.log(1 - r2cs / shrinkage))
    s2 = r2cs / (r2cs + delta * mx)
    n2 = p / ((s2 - 1) * math.log(1 - r2cs / s2))
    n3 = (1.96 / margin) ** 2 * phi * (1 - phi)
    ns = [math.ceil(n1), math.ceil(n2), math.ceil(n3)]
    n = max(ns)
    return dict(criteria=[
        ("(i) Little overfitting: shrinkage factor >= %.2f" % shrinkage, ns[0]),
        ("(ii) Small optimism in model fit (Nagelkerke R2 difference <= %.2f)" % delta, ns[1]),
        ("(iii) Precise overall risk (margin of error <= %.2f)" % margin, ns[2]),
    ], n=n, events=math.ceil(n * phi), epp=n * phi / p, max_r2=mx, r2cs=r2cs,
        shrinkage_at_n=None)


def epv_rule(prevalence: float, parameters: int, epv: float = 10) -> int:
    return math.ceil(epv * parameters / min(prevalence, 1 - prevalence))


def continuous_rule(parameters: int, r2: float, shrinkage: float = 0.9) -> dict:
    """Approximate guidance for a continuous outcome (Riley et al., Stat Med 2019, criteria i and iii)."""
    p = parameters
    n1 = math.ceil(p / ((shrinkage - 1) * math.log(1 - r2 / shrinkage)))
    n3 = 234 + p
    return dict(criteria=[("(i) Little overfitting: shrinkage >= %.2f (approximate)" % shrinkage, n1),
                          ("(iii) Precise estimate of the residual variance (n >= 234 + parameters)", n3)],
                n=max(n1, n3))
