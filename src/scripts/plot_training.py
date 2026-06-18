"""Generate comparison charts from match logs and training data."""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import MATCH_LOG_FILE


SCENARIO_COLORS = {
    "Minimax(d=3)": "#2196F3",
    "AlphaZero(sims=80)": "#FF5722",
    "Minimax(d=2)": "#4CAF50",
    "AlphaZero(sims=200)": "#9C27B0",
    "AlphaZero(sims=800)": "#FF9800",
    "Minimax(d=1)": "#00BCD4",
    "Minimax(d=0)": "#607D8B",
    "RL Agent": "#E91E63",
    "Draw": "#9E9E9E",
}

OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_matches(csv_path: str = MATCH_LOG_FILE) -> list[dict]:
    if not os.path.isfile(csv_path):
        print(f"Match log not found: {csv_path}")
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def group_by_scenario(rows: list[dict]) -> dict[str, list[dict]]:
    scenarios: dict[str, list[dict]] = defaultdict(list)
    batch_size = 20
    for row in rows:
        key = f"{row['ai1']} vs {row['ai2']}"
        scenarios[key].append(row)

    groups: dict[str, list[dict]] = {}
    for key, group in scenarios.items():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            batch_match_id = batch[0]["match_id"]
            label = f"{key} (run #{i // batch_size + 1})"
            groups[label] = batch
    return groups


def plot_win_rate_comparison(scenario_groups: dict[str, list[dict]]) -> str:
    labels = []
    mm_wins = []
    az_wins = []
    draws = []

    for name, matches in scenario_groups.items():
        m = sum(1 for r in matches if "Minimax" in r["winner"] and "Draw" not in r["winner"])
        a = sum(1 for r in matches if "AlphaZero" in r["winner"] or "RL Agent" in r["winner"])
        d = sum(1 for r in matches if r["winner"] == "Draw")
        total = len(matches)

        short = name.split(" vs ")[0][:20] if "Minimax" in name.split(" vs ")[0] else name.split(" vs ")[1][:20]
        labels.append(short)
        mm_wins.append(m / total * 100)
        az_wins.append(a / total * 100)
        draws.append(d / total * 100)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, mm_wins, width, label="Minimax wins", color="#2196F3")
    ax.bar(x, az_wins, width, label="AlphaZero/RL wins", color="#FF5722")
    ax.bar(x + width, draws, width, label="Draws", color="#9E9E9E")

    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win Rate Comparison by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 110)

    for i in range(len(labels)):
        if mm_wins[i] > 0:
            ax.text(i - width, mm_wins[i] + 1, f"{mm_wins[i]:.0f}%", ha="center", va="bottom", fontsize=7)
        if az_wins[i] > 0:
            ax.text(i, az_wins[i] + 1, f"{az_wins[i]:.0f}%", ha="center", va="bottom", fontsize=7)
        if draws[i] > 0:
            ax.text(i + width, draws[i] + 1, f"{draws[i]:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "win_rate_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_move_count_distribution(rows: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    move_counts = [int(r["move_count"]) for r in rows if r.get("move_count")]
    ax.hist(move_counts, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(move_counts), color="red", linestyle="--", label=f'Mean: {np.mean(move_counts):.1f}')
    ax.set_xlabel("Move count")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Match Lengths (all runs)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "move_count_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_thinking_time(rows: list[dict]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    stages = [("Early", "avg_time_early"), ("Mid", "avg_time_mid"), ("End", "avg_time_end")]

    for ax, (stage_name, field) in zip(axes, stages):
        times = []
        for r in rows:
            val = float(r.get(field, 0))
            if val > 0:
                times.append(val)
        if times:
            ax.hist(times, bins=15, alpha=0.7, color="#2196F3", edgecolor="white")
            ax.axvline(np.mean(times), color="red", linestyle="--", label=f'Mean: {np.mean(times):.2f}s')
        ax.set_title(f"{stage_name} Game")
        ax.set_xlabel("Time per move (s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "thinking_time_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_scenario_detail(scenario_name: str, matches: list[dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    match_ids = [int(r["match_id"]) for r in matches]
    m_wins = [1 if "Minimax" in r["winner"] and "Draw" not in r["winner"] else 0 for r in matches]
    a_wins = [1 if ("AlphaZero" in r["winner"] or "RL Agent" in r["winner"]) else 0 for r in matches]
    d_raws = [1 if r["winner"] == "Draw" else 0 for r in matches]

    mm_cum = np.cumsum(m_wins)
    az_cum = np.cumsum(a_wins)
    ax1.plot(match_ids, mm_cum, label="Minimax wins", color="#2196F3", marker="o", markersize=4)
    ax1.plot(match_ids, az_cum, label="AlphaZero/RL wins", color="#FF5722", marker="s", markersize=4)
    ax1.set_xlabel("Match")
    ax1.set_ylabel("Cumulative wins")
    ax1.set_title(f"{scenario_name[:50]}...")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    move_counts = [int(r["move_count"]) for r in matches]
    colors = ["#2196F3" if w else "#FF5722" if a else "#9E9E9E" for w, a, d in zip(m_wins, a_wins, d_raws)]
    ax2.bar(match_ids, move_counts, color=colors, alpha=0.8)
    ax2.axhline(np.mean(move_counts), color="black", linestyle="--", label=f'Avg: {np.mean(move_counts):.1f}')
    ax2.set_xlabel("Match")
    ax2.set_ylabel("Move count")
    ax2.set_title("Move Count per Match")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Minimax win"),
        Patch(facecolor="#FF5722", label="AlphaZero win"),
        Patch(facecolor="#9E9E9E", label="Draw"),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc="upper left")

    fig.tight_layout()
    safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")[:60]
    path = os.path.join(OUTPUT_DIR, f"scenario_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison charts from match logs")
    parser.add_argument("--input", type=str, default=MATCH_LOG_FILE, help="Path to matches.csv")
    parser.add_argument("--scenario", type=str, default=None, help="Plot specific scenario (substring match)")
    args = parser.parse_args()

    rows = load_matches(args.input)
    if not rows:
        print("No match data found.")
        return

    print(f"Loaded {len(rows)} match records from {args.input}")

    scenario_groups = group_by_scenario(rows)

    if args.scenario:
        target = [k for k in scenario_groups if args.scenario in k]
        if not target:
            print(f"No scenario matching '{args.scenario}'. Available:")
            for k in scenario_groups:
                print(f"  {k}")
            return
        for name in target:
            plot_scenario_detail(name, scenario_groups[name])
    else:
        plot_win_rate_comparison(scenario_groups)
        plot_move_count_distribution(rows)
        plot_thinking_time(rows)
        for name, matches in scenario_groups.items():
            if "Minimax(d=3) vs AlphaZero(sims=80)" in name:
                plot_scenario_detail(name, matches)

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
