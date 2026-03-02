from math import log, sqrt, exp, isfinite
from scipy.stats import norm

"""
Black-Scholes European option pricing.

Provides bs_call and bs_put functions.
"""



def _validate_inputs(S0, K, r, sigma, T):
    try:
        S0 = float(S0)
        K = float(K)
        r = float(r)
        sigma = float(sigma)
        T = float(T)
    except (TypeError, ValueError):
        raise ValueError("All inputs must be numeric")

    if not (S0 > 0):
        raise ValueError("S0 must be > 0")
    if not (K > 0):
        raise ValueError("K must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if T < 0:
        raise ValueError("T must be >= 0")
    if not (isfinite(r) and isfinite(sigma) and isfinite(T) and isfinite(S0) and isfinite(K)):
        raise ValueError("r, sigma, T, S0, and K must be finite numbers")

    return S0, K, r, sigma, T


def bs_call(S0, K, r, sigma, T):
    """
    Price a European call option using the Black-Scholes formula.

    Parameters
    - S0: spot price (float, > 0)
    - K: strike price (float, > 0)
    - r: continuous risk-free interest rate (float)
    - sigma: volatility (float, >= 0)
    - T: time to maturity in years (float, >= 0)

    Special cases
    - If T == 0: returns intrinsic value max(S0 - K, 0)
    - If sigma == 0 and T > 0: returns discounted intrinsic value
      max(S0 - K * exp(-r * T), 0)

    Returns
    - price as float
    """
    S0, K, r, sigma, T = _validate_inputs(S0, K, r, sigma, T)

    # At maturity: intrinsic value
    if T == 0:
        return float(max(S0 - K, 0.0))

    # Zero volatility: deterministic growth under risk-neutral measure
    if sigma == 0:
        return float(max(S0 - K * exp(-r * T), 0.0))

    # Standard Black-Scholes formula
    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return float(price)


def bs_put(S0, K, r, sigma, T):
    """
    Price a European put option using the Black-Scholes formula.

    Parameters
    - S0: spot price (float, > 0)
    - K: strike price (float, > 0)
    - r: continuous risk-free interest rate (float)
    - sigma: volatility (float, >= 0)
    - T: time to maturity in years (float, >= 0)

    Special cases
    - If T == 0: returns intrinsic value max(K - S0, 0)
    - If sigma == 0 and T > 0: returns discounted intrinsic value
      max(K * exp(-r * T) - S0, 0)

    Returns
    - price as float
    """
    S0, K, r, sigma, T = _validate_inputs(S0, K, r, sigma, T)

    # At maturity: intrinsic value
    if T == 0:
        return float(max(K - S0, 0.0))

    # Zero volatility: deterministic growth under risk-neutral measure
    if sigma == 0:
        return float(max(K * exp(-r * T) - S0, 0.0))

    # Standard Black-Scholes formula (put-call parity equivalent form)
    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return float(price)