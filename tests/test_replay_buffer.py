import os
import tempfile
import numpy as np
from src.utils.replay_buffer import ReplayBuffer


def test_push_and_len() -> None:
    buffer = ReplayBuffer(capacity=100)
    assert len(buffer) == 0
    buffer.push(np.zeros((3, 9, 9)), np.zeros(81), 1.0)
    assert len(buffer) == 1


def test_sample() -> None:
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        buffer.push(np.zeros((3, 9, 9)), np.zeros(81), float(i))
    states, policies, rewards = buffer.sample(5)
    assert states.shape == (5, 3, 9, 9)
    assert policies.shape == (5, 81)
    assert rewards.shape == (5,)


def test_capacity_overflow() -> None:
    buffer = ReplayBuffer(capacity=5)
    for i in range(10):
        buffer.push(np.zeros((3, 9, 9)), np.zeros(81), float(i))
    assert len(buffer) == 5


def test_save_and_load() -> None:
    buffer = ReplayBuffer(capacity=100)
    for i in range(5):
        buffer.push(np.zeros((3, 9, 9)), np.zeros(81), float(i))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "buffer.npz")
        buffer.save(path)

        buffer2 = ReplayBuffer(capacity=100)
        buffer2.load(path)
        assert len(buffer2) == 5
        assert np.allclose(buffer2.rewards[0], 0.0)
        assert np.allclose(buffer2.rewards[4], 4.0)
