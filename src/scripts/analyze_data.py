"""Analyze and report statistics from match logs, replay buffer, and training data."""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import MATCH_LOG_FILE, REPLAY_LOG_FILE
from src.game.constants import BOARD_SIZE


def analyze_matches(csv_path: str = MATCH_LOG_FILE) -> dict:
    if not os.path.isfile(csv_path):
        return {"error": f"File not found: {csv_path}"}

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    stats: dict = {
        "total_matches": len(rows),
        "scenarios": defaultdict(lambda: {"count": 0, "wins": Counter(), "move_counts": [], "times_by_stage": defaultdict(list)}),
    }

    for r in rows:
        key = f"{r['ai1']} vs {r['ai2']}"
        scene = stats["scenarios"][key]
        scene["count"] += 1
        scene["wins"][r["winner"]] += 1
        scene["move_counts"].append(int(r["move_count"]))
        for stage in ["early", "mid", "end"]:
            val = float(r.get(f"avg_time_{stage}", 0))
            if val > 0:
                scene["times_by_stage"][stage].append(val)

    result = {
        "total_matches": stats["total_matches"],
        "scenarios": [],
    }

    for name, scene in sorted(stats["scenarios"].items()):
        moves = scene["move_counts"]
        win_pct = {}
        for w, c in scene["wins"].items():
            win_pct[w] = f"{c / scene['count'] * 100:.1f}%"

        avg_times = {}
        for stage, vals in scene["times_by_stage"].items():
            avg_times[stage] = f"{np.mean(vals):.4f}s" if vals else "N/A"

        result["scenarios"].append({
            "name": name,
            "matches": scene["count"],
            "wins": dict(scene["wins"]),
            "win_pct": win_pct,
            "move_count_avg": f"{np.mean(moves):.1f}",
            "move_count_std": f"{np.std(moves):.1f}",
            "move_count_min": min(moves),
            "move_count_max": max(moves),
            "avg_times_by_stage": avg_times,
        })

    return result


def analyze_checkpoint(path: str = "models/checkpoint.json") -> dict:
    if not os.path.isfile(path):
        return {"error": f"File not found: {path}"}
    with open(path) as f:
        return json.load(f)


def estimate_dataset_size(buffer_path: str = "models/rl_agent_buffer.npz") -> dict:
    if not os.path.isfile(buffer_path):
        return {"error": f"Buffer not found: {buffer_path}"}
    data = np.load(buffer_path, allow_pickle=True)
    info = {
        "states": data.get("states", np.array([])).shape if "states" in data else 0,
        "policies": data.get("policies", np.array([])).shape if "policies" in data else 0,
        "rewards": data.get("rewards", np.array([])).shape if "rewards" in data else 0,
    }
    if len(info["states"]) > 1:
        n_samples = info["states"][0]
        reward_dist = Counter(data["rewards"].tolist())
        info["total_samples"] = n_samples
        info["reward_distribution"] = {f"{k:.1f}": v for k, v in sorted(reward_dist.items())}
        info["state_shape"] = list(info["states"][1:])
        info["memory_estimate_mb"] = f"{data['states'].nbytes / 1e6:.1f}"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and report dataset/match statistics")
    parser.add_argument("--matches", type=str, default=MATCH_LOG_FILE, help="Path to matches.csv")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoint.json", help="Path to checkpoint.json")
    parser.add_argument("--buffer", type=str, default="models/rl_agent_buffer.npz", help="Path to replay buffer .npz")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    match_stats = analyze_matches(args.matches)
    checkpoint = analyze_checkpoint(args.checkpoint)
    buffer_info = estimate_dataset_size(args.buffer)

    report = {
        "match_statistics": match_stats,
        "training_checkpoint": checkpoint,
        "dataset": buffer_info,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 65)
    print("  GOMOKU AI — DATA & MATCH ANALYSIS REPORT")
    print("=" * 65)

    if "error" in match_stats:
        print(f"\n[!] Match analysis: {match_stats['error']}")
    else:
        print(f"\n--- MATCH STATISTICS ({match_stats['total_matches']} total) ---")
        for s in match_stats["scenarios"]:
            print(f"\n  Scenario: {s['name']}")
            print(f"    Matches: {s['matches']}")
            print(f"    Wins: {s['wins']}")
            print(f"    Win %: {s['win_pct']}")
            print(f"    Moves: avg={s['move_count_avg']}, std={s['move_count_std']}, "
                  f"min={s['move_count_min']}, max={s['move_count_max']}")
            print(f"    Avg thinking time: {s['avg_times_by_stage']}")

    if "error" in checkpoint:
        print(f"\n[!] Checkpoint: {checkpoint['error']}")
    else:
        print(f"\n--- TRAINING CHECKPOINT ---")
        print(f"  Episodes done: {checkpoint.get('episodes_done', 'N/A')}")
        print(f"  Win counts: {checkpoint.get('win_counts', {})}")
        print(f"  MCTS sims: {checkpoint.get('mcts_sims', 'N/A')}")
        print(f"  c_puct: {checkpoint.get('c_puct', 'N/A')}")

    if "error" in buffer_info:
        print(f"\n[!] Buffer: {buffer_info['error']}")
    else:
        print(f"\n--- REPLAY BUFFER DATASET ---")
        print(f"  Total samples: {buffer_info.get('total_samples', 'N/A')}")
        print(f"  State shape: {buffer_info.get('state_shape', 'N/A')}")
        print(f"  Reward distribution: {buffer_info.get('reward_distribution', {})}")
        print(f"  Memory estimate: {buffer_info.get('memory_estimate_mb', 'N/A')} MB")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
