from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.game.constants import BOARD_SIZE, EMPTY, O, X
from src.game.rules import is_draw, is_win


def _other_player(player: int) -> int:
    return O if player == X else X


def _encode_board(grid: np.ndarray, player: int) -> np.ndarray:
    state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0] = (grid == player).astype(np.float32)
    state[1] = (grid == _other_player(player)).astype(np.float32)
    state[2] = (grid == EMPTY).astype(np.float32)
    return state


@dataclass
class MCTSNode:
    player: int
    parent: MCTSNode | None = None
    prior: float = 0.0
    move_from_parent: tuple[int, int] | None = None
    children: dict[tuple[int, int], MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search with PUCT."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: str = "cpu",
        num_simulations: int = 200,
        c_puct: float = 1.5,
    ) -> None:
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(
        self,
        root_state: np.ndarray,
        player: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        root = MCTSNode(player=player)
        # Root expansion
        self._evaluate_and_expand(root, root_state)

        for _ in range(self.num_simulations):
            node = root
            path: list[MCTSNode] = [node]
            current_state = root_state.copy()

            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                current_state[move[0], move[1]] = _other_player(node.player)
                path.append(node)

            value = self._evaluate_and_expand(node, current_state)
            self._backpropagate(path, value)

        return self._build_policy(root, temperature)

    def _select_child(self, node: MCTSNode) -> tuple[tuple[int, int], MCTSNode]:
        moves = list(node.children.keys())
        children = list(node.children.values())
        
        visit_counts = np.array([c.visit_count for c in children])
        priors = np.array([c.prior for c in children])
        # Child Q-value is from other player's perspective, so we negate it
        q_values = np.array([-c.q_value for c in children])
        
        exploration = self.c_puct * priors * np.sqrt(node.visit_count) / (1 + visit_counts)
        scores = q_values + exploration
        
        best_idx = np.argmax(scores)
        return moves[best_idx], children[best_idx]

    def _evaluate_and_expand(self, node: MCTSNode, state: np.ndarray) -> float:
        terminal_value = self._terminal_value(node, state)
        if terminal_value is not None:
            return terminal_value

        policy, value = self._predict(state, node.player)
        valid_mask = (state.flatten() == EMPTY).astype(np.float32)
        masked_policy = policy * valid_mask
        total = float(masked_policy.sum())

        if total <= 0:
            masked_policy = valid_mask / valid_mask.sum()
        else:
            masked_policy /= total

        node.is_expanded = True
        for idx in np.where(valid_mask > 0)[0]:
            r, c = divmod(int(idx), BOARD_SIZE)
            node.children[(r, c)] = MCTSNode(
                player=_other_player(node.player),
                parent=node,
                prior=float(masked_policy[idx]),
                move_from_parent=(r, c),
            )

        return value

    def _predict(self, grid: np.ndarray, player: int) -> tuple[np.ndarray, float]:
        state = _encode_board(grid, player)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

        policy = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy().astype(np.float32)
        value_scalar = float(value.squeeze(0).item())
        return policy, value_scalar

    def _terminal_value(self, node: MCTSNode, state: np.ndarray) -> float | None:
        if node.move_from_parent is not None:
            prev_player = _other_player(node.player)
            if is_win(state, prev_player, last_move=node.move_from_parent):
                return -1.0

        if is_draw(state):
            return 0.0

        return None

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _build_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)

        if not root.children:
            valid = np.where(root.state.flatten() == EMPTY)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            return pi

        for move, child in root.children.items():
            idx = move[0] * BOARD_SIZE + move[1]
            pi[idx] = float(child.visit_count)

        if temperature <= 0:
            out = np.zeros_like(pi)
            out[int(np.argmax(pi))] = 1.0
            return out

        pi = np.where(pi > 0, np.power(pi, 1.0 / temperature), 0.0)
        total = float(pi.sum())
        if total <= 0:
            valid = np.where(root.state.flatten() == EMPTY)[0]
            if len(valid) > 0:
                pi[valid] = 1.0 / len(valid)
            return pi

        return pi / total
