import os
import json
import csv
import tempfile
import numpy as np
import pytest
from src.utils.logger import log_match, log_game_replay, load_logs, load_replays


@pytest.fixture(autouse=True)
def isolated_logs(monkeypatch, tmp_path):
    log_dir = str(tmp_path / "logs")
    monkeypatch.setattr("src.utils.logger.LOG_DIR", log_dir)
    monkeypatch.setattr("src.utils.logger.MATCH_LOG_FILE", os.path.join(log_dir, "matches.csv"))
    monkeypatch.setattr("src.utils.logger.REPLAY_LOG_FILE", os.path.join(log_dir, "replays.jsonl"))
    yield


def test_log_match_creates_csv():
    log_match(
        match_id=1,
        ai1_name="A",
        ai2_name="B",
        winner=1,
        move_count=5,
        times_by_stage={"early": [0.1], "mid": [0.2], "end": [0.3]},
    )
    logs = load_logs()
    assert len(logs) == 1
    assert logs[0]["winner"] == "A"
    assert logs[0]["move_count"] == "5"


def test_log_match_draw():
    log_match(
        match_id=2,
        ai1_name="A",
        ai2_name="B",
        winner=None,
        move_count=81,
        times_by_stage={"early": [], "mid": [], "end": []},
    )
    logs = load_logs()
    assert logs[-1]["winner"] == "Draw"


def test_log_match_empty_stages():
    log_match(
        match_id=3,
        ai1_name="X",
        ai2_name="Y",
        winner=2,
        move_count=3,
        times_by_stage={"early": [], "mid": [], "end": []},
    )
    logs = load_logs()
    assert logs[-1]["winner"] == "Y"


def test_log_game_replay_creates_jsonl():
    log_game_replay(match_id=1, moves=[(1, 0, 0), (2, 1, 1)], winner=1)
    replays = load_replays()
    assert len(replays) == 1
    assert replays[0]["winner"] == "X"
    assert len(replays[0]["moves"]) == 2


def test_log_game_replay_draw():
    log_game_replay(match_id=2, moves=[], winner=None)
    replays = load_replays()
    assert replays[-1]["winner"] == "Draw"


def test_log_game_replay_numpy_int_types():
    log_game_replay(
        match_id=np.int64(3),
        moves=[(np.int64(1), np.int64(0), np.int64(0))],
        winner=np.int64(1),
    )
    replays = load_replays()
    assert replays[-1]["match_id"] == 3
    assert replays[-1]["moves"][0]["player"] == 1
    assert replays[-1]["moves"][0]["row"] == 0
    assert replays[-1]["moves"][0]["col"] == 0
    assert replays[-1]["winner"] == "X"


def test_load_logs_empty():
    assert load_logs() == []


def test_load_replays_empty():
    assert load_replays() == []
