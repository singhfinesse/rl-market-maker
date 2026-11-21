"""Quick sanity tests. Run with:  python -m pytest -q"""
import numpy as np
from src.market import Market, MarketConfig
from src.avellaneda import AvellanedaStoikov, SymmetricStrategy, as_spread


def test_wealth_accounting_adds_up():
    """Wealth change must equal cash change plus inventory revaluation."""
    m = Market(MarketConfig(), n_envs=50, seed=1)
    for _ in range(50):
        s_before, q_before, cash_before = m.s.copy(), m.q.copy(), m.cash.copy()
        d_wealth = m.step(np.full(50, 0.5), np.full(50, 0.5))
        expected = (m.cash - cash_before) + m.q * m.s - q_before * s_before
        assert np.allclose(d_wealth, expected)


def test_price_path_ignores_the_agent():
    """Different quotes must not change the price path (needed for fair comparisons)."""
    a, b = Market(MarketConfig(messy=True), 20, seed=5), Market(MarketConfig(messy=True), 20, seed=5)
    for _ in range(100):
        a.step(np.full(20, 0.1), np.full(20, 0.1))
        b.step(np.full(20, 3.0), np.full(20, 0.5))
    assert np.allclose(a.s, b.s)


def test_tighter_quotes_fill_more():
    cfg = MarketConfig()
    def fills(offset):
        m = Market(cfg, 2000, seed=3)
        m.step(np.full(2000, offset), np.full(2000, 100.0))   # only the bid can fill
        return m.q.sum()
    assert fills(0.1) > fills(2.0)


def test_avellaneda_skews_against_inventory():
    """If we are long, the bid should move away from mid and the ask closer."""
    cfg = MarketConfig()
    strat = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5)
    m = Market(cfg, 3, seed=0)
    m.q = np.array([-5.0, 0.0, 5.0])
    d_bid, d_ask = strat.quote(m)
    assert d_bid[2] > d_bid[1] > d_bid[0]      # long -> bid far below mid
    assert d_ask[2] < d_ask[1] < d_ask[0]      # long -> ask close to mid
    assert np.isclose(d_bid[1], d_ask[1])      # flat -> symmetric


def test_spread_widens_with_volatility():
    assert as_spread(0.1, 4.0, 1.5, 1.0) > as_spread(0.1, 1.0, 1.5, 1.0)


def test_episode_length():
    m = Market(MarketConfig(), 4, seed=0)
    steps = 0
    while not m.done:
        m.step(np.ones(4), np.ones(4)); steps += 1
    assert steps == MarketConfig().n_steps
