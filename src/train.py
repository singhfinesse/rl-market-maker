"""
Train a PPO market maker.

Usage (from the project root):
    python -m src.train --regime clean --steps 1000000
    python -m src.train --regime messy --steps 1000000

PPO (Proximal Policy Optimization) is a standard, stable RL algorithm. We use
the implementation from Stable-Baselines3 rather than writing it ourselves.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .market import MarketConfig, calibrate_sigma
from .env import MarketMakerEnv


def make_cfg(regime: str) -> MarketConfig:
    return MarketConfig(messy=(regime == "messy"))


def train(regime: str, total_steps: int, seed: int = 0, n_envs: int = 8, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = make_cfg(regime)
    sigma = calibrate_sigma(cfg)          # market's overall volatility, for the risk penalty
    print(f"[{regime}] overall volatility = {sigma:.3f}")

    def make_env(rank):
        def _init():
            env = MarketMakerEnv(cfg, penalty_sigma=sigma, seed=seed * 1000 + rank)
            return Monitor(env, filename=os.path.join(out_dir, f"monitor_{regime}_{rank}.csv"))
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        n_steps=256,            # steps collected per env before each update
        batch_size=256,
        n_epochs=10,
        gamma=0.999,            # discount: a session is short, so value the future almost fully
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=total_steps, progress_bar=False)
    path = os.path.join(out_dir, f"ppo_{regime}")
    model.save(path)
    print(f"saved {path}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["clean", "messy"], default="clean")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.regime, args.steps, args.seed)
