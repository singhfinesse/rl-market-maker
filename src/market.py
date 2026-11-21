"""
Simulated market for a single stock.

Two modes
---------
clean : the textbook world assumed by Avellaneda & Stoikov (2008).
        Price = random walk with constant volatility. Orders arrive
        independently of everything else.
messy : a harder world where the textbook assumptions break.
        - volatility flips between a calm and a stormy regime
        - the price occasionally jumps
        - the price has a short-lived drift (momentum), and order flow
          leans WITH that drift. This is "adverse selection": when you
          sell to an eager buyer, the price is about to go up on you.

All episodes run in parallel as numpy arrays (n_envs of them), so evaluating
thousands of episodes takes seconds.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MarketConfig:
    # --- parameters taken from the Avellaneda-Stoikov paper's own simulation ---
    s0: float = 100.0       # starting mid price
    sigma: float = 2.0      # volatility (price units per sqrt(time))
    T: float = 1.0          # length of the trading session
    n_steps: int = 200      # steps per session  -> dt = 0.005
    A: float = 140.0        # base order arrival rate (orders per unit time)
    k: float = 1.5          # how fast fills dry up as quotes move away from mid
    # --- messy-mode parameters ---
    messy: bool = False
    vol_low: float = 1.0
    vol_high: float = 4.0
    p_switch: float = 0.02      # chance per step of flipping volatility regime
    jump_rate: float = 5.0      # expected jumps per unit time
    jump_std: float = 1.0       # typical jump size (price units)
    drift_kappa: float = 10.0   # how fast the drift fades (mean reversion speed)
    drift_std: float = 8.0      # typical size of the drift
    flow_beta: float = 0.5      # how strongly order flow leans with the drift
    # --- quoting limits ---
    max_offset: float = 4.0     # furthest from mid we allow a quote to sit

    @property
    def dt(self) -> float:
        return self.T / self.n_steps


class Market:
    """A batch of independent trading sessions simulated side by side."""

    EMA_RET = 0.2      # smoothing for recent-return signal
    EMA_FLOW = 0.2     # smoothing for recent-order-flow signal
    EMA_VAR = 0.05     # smoothing for realized-volatility estimate (~20 step window)

    def __init__(self, cfg: MarketConfig, n_envs: int = 1, seed: int = 0):
        self.cfg = cfg
        self.n = n_envs
        self.seed(seed)
        self.reset()

    # separate random streams: price paths don't depend on what the agent does,
    # so different strategies can be tested on IDENTICAL price paths (fair test)
    def seed(self, seed: int):
        self.rng_price = np.random.default_rng(seed)
        self.rng_fill = np.random.default_rng(seed + 10_000_019)

    def reset(self):
        c, n = self.cfg, self.n
        self.t = 0
        self.s = np.full(n, c.s0)          # mid price
        self.q = np.zeros(n)               # inventory (shares held)
        self.cash = np.zeros(n)
        self.high_vol = self.rng_price.random(n) < 0.5
        self.mu = (self.rng_price.normal(0, c.drift_std, n) if c.messy else np.zeros(n))
        # running signals the agents can look at
        self.last_ret = np.zeros(n)
        self.ema_ret = np.zeros(n)
        self.ema_flow = np.zeros(n)
        self.ema_var = np.full(n, c.sigma ** 2 * c.dt)
        return self

    # ------------------------------------------------------------------
    @property
    def wealth(self):
        """Cash plus inventory valued at the current mid price."""
        return self.cash + self.q * self.s

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.n_steps

    @property
    def time_left(self) -> float:
        return (self.cfg.n_steps - self.t) * self.cfg.dt

    @property
    def realized_sigma(self):
        """Volatility estimated from the last ~20 price changes."""
        return np.sqrt(self.ema_var / self.cfg.dt)

    # ------------------------------------------------------------------
    def step(self, d_bid, d_ask):
        """
        d_bid / d_ask: how far BELOW / ABOVE the mid price we place our bid / ask.
        Returns the change in wealth over this step.
        """
        c, n, dt = self.cfg, self.n, self.cfg.dt
        d_bid = np.clip(d_bid, 0.0, c.max_offset)
        d_ask = np.clip(d_ask, 0.0, c.max_offset)
        w_before = self.wealth

        # 1) Do our quotes get filled?  Arrival rate falls exponentially with distance.
        lam_bid = c.A * np.exp(-c.k * d_bid)    # sellers arriving to hit our bid
        lam_ask = c.A * np.exp(-c.k * d_ask)    # buyers arriving to lift our ask
        if c.messy:
            tilt = np.exp(c.flow_beta * self.mu / c.drift_std)
            lam_ask = lam_ask * tilt            # price drifting up -> more buyers
            lam_bid = lam_bid / tilt            # price drifting down -> more sellers
        fill_bid = self.rng_fill.random(n) < 1 - np.exp(-lam_bid * dt)
        fill_ask = self.rng_fill.random(n) < 1 - np.exp(-lam_ask * dt)

        self.q += fill_bid.astype(float) - fill_ask.astype(float)
        self.cash += fill_ask * (self.s + d_ask) - fill_bid * (self.s - d_bid)

        # 2) The mid price moves.
        z = self.rng_price.standard_normal(n)
        sig = np.where(self.high_vol, c.vol_high, c.vol_low) if c.messy else c.sigma
        ds = sig * np.sqrt(dt) * z
        if c.messy:
            ds = ds + self.mu * dt
            jumped = self.rng_price.random(n) < c.jump_rate * dt
            ds = ds + jumped * self.rng_price.normal(0, c.jump_std, n)
            self.mu += (-c.drift_kappa * self.mu * dt
                        + c.drift_std * np.sqrt(2 * c.drift_kappa * dt)
                        * self.rng_price.standard_normal(n))
            flip = self.rng_price.random(n) < c.p_switch
            self.high_vol = np.where(flip, ~self.high_vol, self.high_vol)
        self.s = self.s + ds

        # 3) update the running signals
        self.last_ret = ds
        self.ema_ret = (1 - self.EMA_RET) * self.ema_ret + self.EMA_RET * ds
        self.ema_var = (1 - self.EMA_VAR) * self.ema_var + self.EMA_VAR * ds ** 2
        flow = fill_ask.astype(float) - fill_bid.astype(float)   # +1 = a buyer hit us
        self.ema_flow = (1 - self.EMA_FLOW) * self.ema_flow + self.EMA_FLOW * flow
        self.t += 1
        return self.wealth - w_before

    # ------------------------------------------------------------------
    def features(self):
        """What the RL agent gets to see. Roughly unit-sized numbers, shape (n, 6)."""
        c = self.cfg
        unit = c.sigma * np.sqrt(c.dt)          # a typical one-step price move
        obs = np.stack([
            self.q / 5.0,                                   # inventory
            np.full(self.n, self.time_left / c.T),          # fraction of session left
            self.last_ret / unit,                           # last price change
            self.ema_ret / unit * 2.0,                      # recent trend
            np.sqrt(self.ema_var) / unit,                   # recent volatility
            self.ema_flow * 2.0,                            # recent order flow (+ = buyers)
        ], axis=1)
        return np.clip(obs, -5, 5).astype(np.float32)


def calibrate_sigma(cfg: MarketConfig, n_envs: int = 2000, seed: int = 123) -> float:
    """
    Measure the market's overall volatility empirically (std of one-step price
    change divided by sqrt(dt)). Used to give Avellaneda-Stoikov the TRUE
    long-run volatility, so the baseline is as strong as it can fairly be.
    """
    m = Market(cfg, n_envs, seed)
    changes = []
    while not m.done:
        before = m.s.copy()
        m.step(np.zeros(n_envs), np.zeros(n_envs))
        changes.append(m.s - before)
    return float(np.std(np.concatenate(changes)) / np.sqrt(cfg.dt))
