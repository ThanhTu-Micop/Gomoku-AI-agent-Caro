import numpy as np
import json
import os
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.states: deque[np.ndarray] = deque(maxlen=capacity)
        self.policies: deque[np.ndarray] = deque(maxlen=capacity)
        self.rewards: deque[float] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: np.ndarray, reward: float) -> None:
        self.states.append(state)
        self.policies.append(policy)
        self.rewards.append(reward)

    def extend(
        self,
        states: list[np.ndarray],
        policies: list[np.ndarray],
        rewards: list[float],
    ) -> None:
        for s, p, r in zip(states, policies, rewards):
            self.push(s, p, r)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.states) < batch_size:
            batch_size = len(self.states)
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        states = np.array([self.states[i] for i in indices])
        policies = np.array([self.policies[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        return states, policies, rewards

    def __len__(self) -> int:
        return len(self.states)

    def is_full(self) -> bool:
        return len(self.states) == self.states.maxlen

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            states=np.array(self.states),
            policies=np.array(self.policies),
            rewards=np.array(self.rewards),
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.states = deque(data["states"], maxlen=self.states.maxlen)
        self.policies = deque(data["policies"], maxlen=self.policies.maxlen)
        self.rewards = deque(data["rewards"], maxlen=self.rewards.maxlen)
