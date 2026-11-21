"""
Gymnasium wrapper so any RL library can learn to market make in our simulator.

State  (what the agent sees, 6 numbers):
    inventory, fraction of session left, last price change, recent trend,
    recent volatility, recent order flow
Action (what the agent does, 2 numbers in [-1, 1]):
    where to put the bid and the ask, mapped to a distance from the mid price
    between 0 and cfg.max_offset
Reward (per step):
    change in wealth  -  inventory penalty
    The penalty is (gamma/2) * sigma^2 * inventory^2 * dt. This is exactly the
    risk cost Avellaneda-Stoikov's formula assumes, so the RL agent and the
    formula are trying to optimise the SAME objective. That makes the comparison fair.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .market import Market, MarketConfig


def action_to_offsets(action, cfg: MarketConfig):
    """Map network output in [-1, 1] to quote distances in [0, max_offset]."""
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    offsets = (a + 1.0) / 2.0 * cfg.max_offset
    return offsets[..., 0], offsets[..., 1]


class MarketMakerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: MarketConfig = None, gamma: float = 0.1,
                 penalty_sigma: float = None, seed: int = 0):
        super().__init__()
        self.cfg = cfg or MarketConfig()
        self.gamma = gamma
        # volatility used inside the inventory penalty (the market's overall vol)
        self.penalty_sigma = penalty_sigma if penalty_sigma is not None else self.cfg.sigma
        self.market = Market(self.cfg, n_envs=1, seed=seed)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.market.seed(seed)
        self.market.reset()
        return self.market.features()[0], {}

    def step(self, action):
        d_bid, d_ask = action_to_offsets(action, self.cfg)
        d_wealth = self.market.step(np.array([d_bid]), np.array([d_ask]))[0]
        q = self.market.q[0]
        penalty = 0.5 * self.gamma * self.penalty_sigma ** 2 * q ** 2 * self.cfg.dt
        reward = float(d_wealth - penalty)
        terminated = self.market.done
        info = {"wealth": float(self.market.wealth[0]), "inventory": float(q)}
        return self.market.features()[0], reward, terminated, False, info
