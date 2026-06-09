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
    virtual_loss: int = 0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        return self.value_sum / effective_visits


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
        batch_size: int = 16,
    ) -> np.ndarray:
        root = MCTSNode(player=player)
        self._evaluate_and_expand_batch([root], [root_state])

        num_batches = max(1, self.num_simulations // batch_size)
        
        with torch.inference_mode():
            for _ in range(num_batches):
                leaves: list[MCTSNode] = []
                leaf_states: list[np.ndarray] = []
                paths: list[list[MCTSNode]] = []

                for _ in range(batch_size):
                    node = root
                    path = [node]
                    current_state = root_state.copy()

                    while node.is_expanded and node.children:
                        move, node = self._select_child(node)
                        current_state[move[0], move[1]] = _other_player(node.player)
                        path.append(node)
                        node.virtual_loss += 3
                    
                    leaves.append(node)
                    leaf_states.append(current_state)
                    paths.append(path)

                values = self._evaluate_and_expand_batch(leaves, leaf_states)
                
                for path, value in zip(paths, values):
                    for node in path[1:]:
                        node.virtual_loss -= 3
                    self._backpropagate(path, value)

        return self._build_policy(root, temperature)

    def _select_child(self, node: MCTSNode) -> tuple[tuple[int, int], MCTSNode]:
        moves = list(node.children.keys())
        children = list(node.children.values())
        
        v_counts = np.array([c.visit_count + c.virtual_loss for c in children])
        priors = np.array([c.prior for c in children])
        q_values = np.array([-c.q_value for c in children])
        
        exploration = self.c_puct * priors * np.sqrt(node.visit_count + node.virtual_loss + 1) / (1 + v_counts)
        scores = q_values + exploration
        
        best_idx = np.argmax(scores)
        return moves[best_idx], children[best_idx]

    def _evaluate_and_expand_batch(self, nodes: list[MCTSNode], states: list[np.ndarray]) -> list[float]:
        results: list[float] = []
        to_predict_indices: list[int] = []
        to_predict_states: list[np.ndarray] = []
        to_predict_players: list[int] = []

        for i, (node, state) in enumerate(zip(nodes, states)):
            if node.move_from_parent is not None:
                prev_player = _other_player(node.player)
                if is_win(state, prev_player, last_move=node.move_from_parent):
                    results.append(-1.0)
                    continue
            if is_draw(state):
                results.append(0.0)
                continue
            
            results.append(0.0)
            to_predict_indices.append(i)
            to_predict_states.append(state)
            to_predict_players.append(node.player)

        if to_predict_states:
            policies, values = self._predict_batch(to_predict_states, to_predict_players)
            for idx, policy, value in zip(to_predict_indices, policies, values):
                node = nodes[idx]
                state = states[idx]
                
                if not node.is_expanded:
                    valid_mask = (state.flatten() == EMPTY).astype(np.float32)
                    masked_policy = policy * valid_mask
                    total = float(masked_policy.sum())
                    if total <= 0:
                        masked_policy = valid_mask / valid_mask.sum()
                    else:
                        masked_policy /= total

                    node.is_expanded = True
                    for move_idx in np.where(valid_mask > 0)[0]:
                        r, c = divmod(int(move_idx), BOARD_SIZE)
                        node.children[(r, c)] = MCTSNode(
                            player=_other_player(node.player),
                            parent=node,
                            prior=float(masked_policy[move_idx]),
                            move_from_parent=(r, c),
                        )
                results[idx] = value

        return results

    def _predict_batch(self, grids: list[np.ndarray], players: list[int]) -> tuple[np.ndarray, np.ndarray]:
        batch_states = np.stack([_encode_board(g, p) for g, p in zip(grids, players)])
        state_tensor = torch.as_tensor(batch_states, dtype=torch.float32, device=self.device)

        policy_logits, values = self.network(state_tensor)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        value_scalars = values.squeeze(1).cpu().numpy()
        return policies, value_scalars

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
            return pi

        for move, child in root.children.items():
            idx = move[0] * BOARD_SIZE + move[1]
            pi[idx] = float(child.visit_count)

        if temperature <= 0:
            out = np.zeros_like(pi)
            out[int(np.argmax(pi))] = 1.0
            return out

        pi = np.power(pi, 1.0 / temperature)
        return pi / pi.sum()
