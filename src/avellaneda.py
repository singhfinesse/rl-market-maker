"""
Hand-built market-making strategies (the benchmarks the RL agent must beat).

Every strategy answers the same question each step:
    "How far below the mid price should my bid sit, and how far above my ask?"
and returns (d_bid, d_ask).

1. SymmetricStrategy       - quote the same distance on both sides, ignore inventory
2. AvellanedaStoikov       - the 2008 formula, given the market's true volatility
3. AvellanedaStoikov (adaptive=True) - same formula, but volatility is estimated
                             from the last ~20 price changes instead of being known
"""
import numpy as np
from .market import Market


def as_spread(gamma: float, sigma: float, k: float, time_left):
    """Total optimal spread between bid and ask (Avellaneda-Stoikov)."""
    return gamma * sigma ** 2 * time_left + (2.0 / gamma) * np.log(1.0 + gamma / k)


class AvellanedaStoikov:
    def __init__(self, gamma=0.1, sigma=None, k=1.5, adaptive=False):
        self.gamma, self.sigma, self.k, self.adaptive = gamma, sigma, k, adaptive
        self.name = "Avellaneda-Stoikov (adaptive vol)" if adaptive else "Avellaneda-Stoikov"

    def quote(self, m: Market):
        tau = m.time_left                                   # time remaining
        sigma = m.realized_sigma if self.adaptive else self.sigma
        # reservation price: our personal "fair value", pushed down when we hold
        # inventory (we want to sell) and up when we are short (we want to buy)
        shift = m.q * self.gamma * sigma ** 2 * tau         # = mid - reservation price
        half_spread = as_spread(self.gamma, sigma, self.k, tau) / 2.0
        d_bid = shift + half_spread                          # distance of bid below mid
        d_ask = half_spread - shift                          # distance of ask above mid
        return d_bid, d_ask


class SymmetricStrategy:
    """
    Quotes a fixed distance either side of the mid price, never reacting to
    inventory. Its spread is set to the AVERAGE spread Avellaneda-Stoikov uses,
    so the comparison isolates the value of inventory-skewing.
    """
    name = "Symmetric (fixed spread)"

    def __init__(self, gamma=0.1, sigma=2.0, k=1.5, T=1.0):
        avg_spread = gamma * sigma ** 2 * T / 2.0 + (2.0 / gamma) * np.log(1.0 + gamma / k)
        self.half = avg_spread / 2.0

    def quote(self, m: Market):
        full = np.full(m.n, self.half)
        return full, full
