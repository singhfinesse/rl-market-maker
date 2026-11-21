"""
Head-to-head comparison of all strategies on thousands of unseen sessions.

Usage (from the project root, after training):
    python -m src.evaluate

Fairness measures
-----------------
* Every strategy trades the SAME price paths and the SAME random fill draws
  (common random numbers), so differences come from the strategy, not luck.
* The evaluation seed is different from every training seed.
* Avellaneda-Stoikov is given the market's true overall volatility.
* The headline "objective" is exactly what the RL agent was trained to
  maximise: profit minus the inventory risk penalty.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .market import Market, MarketConfig, calibrate_sigma
from .avellaneda import AvellanedaStoikov, SymmetricStrategy
from .env import action_to_offsets

GAMMA = 0.1
N_EPISODES = 5000
EVAL_SEED = 2024
RESULTS = "results"
MODELS = "models"

COLORS = {
    "Symmetric (fixed spread)": "#9aa0a6",
    "Avellaneda-Stoikov": "#1f77b4",
    "Avellaneda-Stoikov (adaptive vol)": "#17becf",
    "PPO (RL agent)": "#d62728",
    "PPO (trained on clean market)": "#ff9896",
}


class PPOPolicy:
    """Wraps a trained PPO model so it quotes like the other strategies."""
    def __init__(self, path, cfg, name="PPO (RL agent)"):
        self.model = PPO.load(path, device="cpu")
        self.cfg, self.name = cfg, name

    def quote(self, m: Market):
        action, _ = self.model.predict(m.features(), deterministic=True)
        return action_to_offsets(action, self.cfg)


def run_sessions(cfg, strategy, sigma_pen, n=N_EPISODES, seed=EVAL_SEED):
    """Trade n sessions in parallel. Returns per-session results and inventory paths."""
    m = Market(cfg, n, seed)
    q_path = [m.q.copy()]
    objective = np.zeros(n)
    while not m.done:
        d_bid, d_ask = strategy.quote(m)
        d_wealth = m.step(d_bid, d_ask)
        penalty = 0.5 * GAMMA * sigma_pen ** 2 * m.q ** 2 * cfg.dt
        objective += d_wealth - penalty
        q_path.append(m.q.copy())
    return {"pnl": m.wealth.copy(), "final_q": m.q.copy(),
            "objective": objective, "q_path": np.array(q_path)}


def trace_session(cfg, strategy, seed):
    """Record one session in detail (price, quotes, inventory) for plotting."""
    m = Market(cfg, 1, seed)
    rows = []
    while not m.done:
        d_bid, d_ask = strategy.quote(m)
        d_bid = np.clip(d_bid, 0, cfg.max_offset)
        d_ask = np.clip(d_ask, 0, cfg.max_offset)
        rows.append((m.t * cfg.dt, m.s[0], m.s[0] - d_bid[0], m.s[0] + d_ask[0], m.q[0]))
        m.step(d_bid, d_ask)
    return pd.DataFrame(rows, columns=["t", "mid", "bid", "ask", "inventory"])


def summarise(results, baseline="Avellaneda-Stoikov", n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    base_obj = results[baseline]["objective"]
    n = len(base_obj)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for name, r in results.items():
        diff = r["objective"] - base_obj                       # paired by session
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "Strategy": name,
            "Mean PnL": r["pnl"].mean(),
            "PnL std": r["pnl"].std(),
            "Sharpe (per session)": r["pnl"].mean() / r["pnl"].std(),
            "Mean |final inventory|": np.abs(r["final_q"]).mean(),
            "Objective (risk-adj. score)": r["objective"].mean(),
            f"Objective vs {baseline.split(' (')[0]}": diff.mean(),
            "95% CI low": lo, "95% CI high": hi,
        })
    return pd.DataFrame(rows)


def evaluate_regime(regime):
    cfg = MarketConfig(messy=(regime == "messy"))
    sigma = calibrate_sigma(cfg)
    print(f"\n=== {regime.upper()} market  (overall volatility {sigma:.2f}) ===")

    strategies = [
        SymmetricStrategy(gamma=GAMMA, sigma=sigma, k=cfg.k, T=cfg.T),
        AvellanedaStoikov(gamma=GAMMA, sigma=sigma, k=cfg.k),
        AvellanedaStoikov(gamma=GAMMA, k=cfg.k, adaptive=True),
        PPOPolicy(os.path.join(MODELS, f"ppo_{regime}"), cfg),
    ]
    if regime == "messy" and os.path.exists(os.path.join(MODELS, "ppo_clean.zip")):
        # robustness check: an agent that only ever saw the clean market
        strategies.append(PPOPolicy(os.path.join(MODELS, "ppo_clean"), cfg,
                                    name="PPO (trained on clean market)"))

    results = {s.name: run_sessions(cfg, s, sigma) for s in strategies}
    table = summarise(results)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(table.round(2).to_string(index=False))
    table.round(3).to_csv(os.path.join(RESULTS, f"summary_{regime}.csv"), index=False)
    return cfg, sigma, strategies, results, table


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def plot_pnl(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, regime in zip(axes, ["clean", "messy"]):
        for name, r in all_res[regime].items():
            ax.hist(r["pnl"], bins=70, histtype="step", linewidth=1.8, density=True,
                    color=COLORS.get(name), label=f"{name}  (mean {r['pnl'].mean():.1f}, std {r['pnl'].std():.1f})")
        ax.set_title(f"Final profit per session - {regime} market")
        ax.set_xlabel("Profit at end of session")
        ax.set_ylabel("Density")
        if regime == "messy":
            ax.set_xlim(-150, 150)   # the symmetric strategy has extreme tails; zoom in
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig1_pnl_distribution.png"), dpi=150)
    plt.close(fig)


def plot_inventory(all_res, cfg):
    t = np.arange(cfg.n_steps + 1) * cfg.dt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, regime in zip(axes, ["clean", "messy"]):
        for name, r in all_res[regime].items():
            ax.plot(t, np.abs(r["q_path"]).mean(axis=1), color=COLORS.get(name), linewidth=1.8, label=name)
        ax.set_title(f"Average absolute inventory over the session - {regime} market")
        ax.set_xlabel("Time in session")
        ax.set_ylabel("Mean |inventory| (shares)")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig2_inventory.png"), dpi=150)
    plt.close(fig)


def plot_training(tables):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, regime in zip(axes, ["clean", "messy"]):
        frames = []
        for f in sorted(os.listdir(MODELS)):
            if f.startswith(f"monitor_{regime}_") and f.endswith(".csv"):
                frames.append(pd.read_csv(os.path.join(MODELS, f), skiprows=1))
        if not frames:
            continue
        df = pd.concat(frames).sort_values("t").reset_index(drop=True)
        ax.plot(df.index, df["r"].rolling(400, min_periods=50).mean(), color="#d62728", label="PPO during training")
        ref = tables[regime].set_index("Strategy")
        ax.axhline(ref.loc["Avellaneda-Stoikov", "Objective (risk-adj. score)"], color="#1f77b4",
                   linestyle="--", label="Avellaneda-Stoikov (evaluation)")
        ax.set_title(f"Learning curve - {regime} market")
        ax.set_xlabel("Training sessions completed")
        ax.set_ylabel("Objective per session (rolling mean)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_example(cfg, strategies, seed=11, regime="messy"):
    pick = [s for s in strategies if s.name in ("Avellaneda-Stoikov", "PPO (RL agent)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for col, s in enumerate(pick):
        tr = trace_session(cfg, s, seed)
        ax, ax2 = axes[0, col], axes[1, col]
        ax.plot(tr.t, tr.mid, color="black", linewidth=1.2, label="mid price")
        ax.step(tr.t, tr.bid, color="#2ca02c", linewidth=0.9, where="post", label="our bid")
        ax.step(tr.t, tr.ask, color="#d62728", linewidth=0.9, where="post", label="our ask")
        ax.set_title(f"{s.name} - one {regime} session")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8, loc="upper left")
        ax2.step(tr.t, tr.inventory, color=COLORS.get(s.name), where="post")
        ax2.axhline(0, color="grey", linewidth=0.6)
        ax2.set_ylabel("Inventory")
        ax2.set_xlabel("Time in session")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_example_session.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    all_res, tables, cfgs, strat_by_regime = {}, {}, {}, {}
    for regime in ["clean", "messy"]:
        cfg, sigma, strategies, results, table = evaluate_regime(regime)
        all_res[regime], tables[regime], cfgs[regime], strat_by_regime[regime] = results, table, cfg, strategies
    plot_pnl(all_res)
    plot_inventory(all_res, cfgs["clean"])
    plot_training(tables)
    plot_example(cfgs["messy"], strat_by_regime["messy"])
    print("\nSaved tables and figures to ./results")


if __name__ == "__main__":
    main()
