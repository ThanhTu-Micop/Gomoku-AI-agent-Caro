import csv
import json
import os
from datetime import datetime
from src.game.constants import X, O


LOG_DIR = "logs"
MATCH_LOG_FILE = os.path.join(LOG_DIR, "matches.csv")
REPLAY_LOG_FILE = os.path.join(LOG_DIR, "replays.jsonl")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def log_match(
    match_id: int,
    ai1_name: str,
    ai2_name: str,
    winner: int | None,
    move_count: int,
    times_by_stage: dict[str, list[float]],
) -> None:
    _ensure_log_dir()

    winner_name = "Draw" if winner is None else (ai1_name if winner == X else ai2_name)

    avg_early = sum(times_by_stage.get("early", [0])) / max(len(times_by_stage.get("early", [1])), 1)
    avg_mid = sum(times_by_stage.get("mid", [0])) / max(len(times_by_stage.get("mid", [1])), 1)
    avg_end = sum(times_by_stage.get("end", [0])) / max(len(times_by_stage.get("end", [1])), 1)

    row = {
        "match_id": match_id,
        "ai1": ai1_name,
        "ai2": ai2_name,
        "winner": winner_name,
        "move_count": move_count,
        "avg_time_early": f"{avg_early:.4f}",
        "avg_time_mid": f"{avg_mid:.4f}",
        "avg_time_end": f"{avg_end:.4f}",
        "timestamp": datetime.now().isoformat(),
    }

    file_exists = os.path.isfile(MATCH_LOG_FILE)

    with open(MATCH_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_logs() -> list[dict]:
    if not os.path.isfile(MATCH_LOG_FILE):
        return []
    with open(MATCH_LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def log_game_replay(
    log_path: str | None = None,
    match_id: int = 0,
    moves: list[tuple[int, int, int]] | None = None,
    winner: int | None = None,
) -> None:
    path = log_path or REPLAY_LOG_FILE
    _ensure_log_dir()
    winner_str = "X" if winner == X else ("O" if winner == O else "Draw")
    record = {
        "match_id": int(match_id),
        "moves": [
            {"player": int(p), "row": int(r), "col": int(c)}
            for p, r, c in (moves or [])
        ],
        "winner": winner_str,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_replays(log_path: str | None = None) -> list[dict]:
    path = log_path or REPLAY_LOG_FILE
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]
