"""
Phase 7 — Training script for the RL placement agent.

Trains a MaskablePPO (sb3-contrib) policy over a suite of circuits derived from
netlists/sample_netlist.json plus the Phase 0 outputs already captured in
netlists/generated/.  A held-out subset (>= 2 circuits) is reserved for
evaluation / benchmarking and is never trained on.

Algorithm choice
----------------
MaskablePPO is used because the action space (a flat grid of placement anchors)
is heavily constrained — most cells overlap an already-placed component or fall
out of bounds.  Action masking lets the agent learn over only the legal anchors
each step, which is far more sample-efficient than penalising illegal actions
after the fact.  (A from-scratch REINFORCE-with-baseline fallback was the
documented alternative; masking integrated cleanly, so PPO is used.)

Policy: MLP with two hidden layers of 256 units (net_arch=[256, 256]).

Hyperparameters RL_TIMESTEPS and RL_LEARNING_RATE are read from .env.

Outputs
-------
models/phase7_rl_placer.zip        — trained policy
outputs/phase7_training_curve.png  — episode-reward learning curve
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from phase1_eda_engine import NetlistParser, InitialPlacer, CircuitGraph
from phase7_rl_placer import PlacementEnv, MODEL_PATH, grid_dims, _OUTPUT_DIR, _MODELS_DIR

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent
_NETLIST_DIR  = _PROJECT_ROOT / "netlists"
_TRAIN_CURVE  = _OUTPUT_DIR / "phase7_training_curve.png"

# Colour palette (mirrors the rest of the project)
_BG, _PANEL_BG, _GRID_C, _TEXT_C, _DIM_C, _EMERALD = (
    "#0f0f1a", "#16162a", "#1e1e3a", "#e0e0ff", "#888899", "#00C97A"
)


# ---------------------------------------------------------------------------
# Circuit suite
# ---------------------------------------------------------------------------

def _load_graph(path: Path) -> CircuitGraph | None:
    """Parse one netlist JSON file into a placed CircuitGraph (None on failure)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        netlist = NetlistParser().parse(raw)
        InitialPlacer(netlist.metadata).place(netlist)
        graph = CircuitGraph.from_netlist(netlist)
    except Exception as exc:                       # malformed netlist — skip
        print(f"  [skip] {path.name}: {exc}")
        return None
    if len(graph.nodes) < 2 or not graph.edges:
        return None
    return graph


def build_suite() -> tuple[list[CircuitGraph], list[CircuitGraph], list[str], list[str]]:
    """Build (train_graphs, eval_graphs, train_names, eval_names).

    The sample netlist plus every generated netlist that fits the .env grid is
    collected and de-duplicated by design name.  The two SMALLEST-named circuits
    after the sample are held out for evaluation (deterministic split).

    Returns:
        Tuple of (train graphs, eval graphs, train names, eval names).
    """
    gw, gh = grid_dims()
    paths = [_NETLIST_DIR / "sample_netlist.json"]
    paths += sorted((_NETLIST_DIR / "generated").glob("*.json"))

    seen_names: set[str] = set()
    graphs: list[tuple[str, CircuitGraph]] = []
    for p in paths:
        g = _load_graph(p)
        if g is None:
            continue
        # Must fit the env grid; skip oversized footprints.
        if any(c.footprint.width > gw or c.footprint.height > gh
               for c in g.nodes.values()):
            continue
        name = g.metadata.name or p.stem
        key = f"{name}:{len(g.nodes)}:{len(g.edges)}"
        if key in seen_names:
            continue
        seen_names.add(key)
        graphs.append((p.stem, g))

    if len(graphs) < 3:
        raise RuntimeError(
            f"Need >= 3 usable circuits to train + hold out 2; found {len(graphs)}."
        )

    # Deterministic split: keep the sample (index 0) in train; hold out the
    # last two distinct circuits for evaluation.
    train = graphs[:-2]
    evalg = graphs[-2:]
    return (
        [g for _, g in train],
        [g for _, g in evalg],
        [n for n, _ in train],
        [n for n, _ in evalg],
    )


# ---------------------------------------------------------------------------
# Reward-curve callback
# ---------------------------------------------------------------------------

def _make_reward_callback():
    """Return a BaseCallback subclass instance that records episode rewards."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardCurveCallback(BaseCallback):
        """Collects per-episode rewards via the Monitor 'episode' info key."""

        def __init__(self) -> None:
            super().__init__()
            self.episode_rewards: list[float] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    self.episode_rewards.append(float(ep["r"]))
            return True

    return RewardCurveCallback()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_curve(rewards: list[float], out_path: Path) -> None:
    """Render the episode-reward learning curve to a PNG."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=_BG)
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_color(_GRID_C)
    ax.tick_params(colors=_DIM_C, labelsize=8)

    if rewards:
        x = np.arange(len(rewards))
        ax.plot(x, rewards, color=_GRID_C, lw=0.6, alpha=0.6, label="Episode reward")
        # Moving average
        w = max(1, len(rewards) // 50)
        if w > 1:
            ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(ma)) + w - 1, ma, color=_EMERALD, lw=2.0,
                    label=f"Moving avg ({w})")
    ax.set_title("Phase 7 — RL Placement Training Curve",
                 color=_TEXT_C, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Episode", color=_DIM_C, fontsize=10)
    ax.set_ylabel("Total episode reward", color=_DIM_C, fontsize=10)
    ax.grid(color=_GRID_C, lw=0.5, alpha=0.5)
    ax.legend(fontsize=9, facecolor=_PANEL_BG, edgecolor=_GRID_C,
              labelcolor=_TEXT_C, framealpha=0.85)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> None:
    """Train MaskablePPO over the circuit suite and save policy + curve."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    timesteps = int(os.getenv("RL_TIMESTEPS", "50000"))
    lr        = float(os.getenv("RL_LEARNING_RATE", "0.0003"))

    train_graphs, eval_graphs, train_names, eval_names = build_suite()
    print(f"[Train] Suite: {len(train_graphs)} train, {len(eval_graphs)} eval.")
    print(f"        Train circuits: {train_names}")
    print(f"        Held-out eval : {eval_names}")
    print(f"[Train] timesteps={timesteps}  learning_rate={lr}")

    def _mask_fn(env: PlacementEnv) -> np.ndarray:
        return env.action_masks()

    # ActionMasker must wrap the PlacementEnv directly (so its mask fn receives
    # the env that defines action_masks); Monitor wraps the result for logging.
    base = PlacementEnv(train_graphs)
    env  = Monitor(ActionMasker(base, _mask_fn))

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        seed=42,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callback = _make_reward_callback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"[Train] Saved policy -> {MODEL_PATH}")

    _plot_curve(callback.episode_rewards, _TRAIN_CURVE)
    print(f"[Train] Saved curve  -> {_TRAIN_CURVE}")
    if callback.episode_rewards:
        tail = callback.episode_rewards[-min(100, len(callback.episode_rewards)):]
        print(f"[Train] Mean reward (last {len(tail)} eps): {np.mean(tail):.3f}")


if __name__ == "__main__":
    train()
