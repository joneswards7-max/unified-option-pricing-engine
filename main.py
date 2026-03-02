from option_pricing.simulation import MarketParams, SimParams, PriceSimulator
from option_pricing.black_scholes import bs_call
import numpy as np

def monte_carlo_call_price(S0, K, r, sigma, T, steps, n_paths, seed=None):
    market = MarketParams(s0=S0, r=r, sigma=sigma)
    sim = SimParams(t=T, steps=steps, n_paths=n_paths, seed=seed)

    paths = PriceSimulator.simulate_gbm_paths(market, sim)

    # Terminal prices
    ST = paths[:, -1]

    # Payoff
    payoff = np.maximum(ST - K, 0.0)

    # Discounted expectation
    price = np.exp(-r * T) * np.mean(payoff)

    return float(price)


if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    steps = 252
    n_paths = 100_000

    mc_price = monte_carlo_call_price(S0, K, r, sigma, T, steps, n_paths, seed=42)
    bs_price = bs_call(S0, K, r, sigma, T)

    print(f"Monte Carlo Price : {mc_price:.6f}")
    print(f"Black-Scholes Price: {bs_price:.6f}")
    print(f"Absolute Error     : {abs(mc_price - bs_price):.6f}")

    path_counts = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
print("\nConvergence Test (Monte Carlo vs Black-Scholes)")
print("Paths\tMC Price\tAbs Error")

for n in path_counts:
    mc = monte_carlo_call_price(S0, K, r, sigma, T, steps, n, seed=42)
    err = abs(mc - bs_price)
    print(f"{n}\t{mc:.6f}\t{err:.6f}")