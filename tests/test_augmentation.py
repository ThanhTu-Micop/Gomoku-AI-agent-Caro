import numpy as np

from src.ai.rl_agent import augment_data


def test_augment_data_returns_8_aligned_pairs() -> None:
    state = np.zeros((3, 9, 9), dtype=np.float32)
    state[0, 0, 0] = 1.0
    policy = np.zeros(81, dtype=np.float32)
    policy[0] = 1.0

    augmented = augment_data(state, policy)
    assert len(augmented) == 8

    for state_aug, policy_aug in augmented:
        piece_positions = np.argwhere(state_aug[0] == 1.0)
        assert len(piece_positions) == 1
        r, c = piece_positions[0]

        policy_2d = policy_aug.reshape(9, 9)
        assert np.isclose(policy_2d.sum(), 1.0)
        assert int(np.argmax(policy_2d)) == int(r * 9 + c)
