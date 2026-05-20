import numpy as np
import random
from src.ai.base import Agent
from src.ai.heuristic import evaluate
from src.game.constants import BOARD_SIZE, EMPTY, X, O
from src.game.rules import is_win, is_draw


class MinimaxAgent(Agent):
    def __init__(self, depth: int = 3) -> None:
        self.max_depth = depth

    def get_move(self, grid: np.ndarray, player: int) -> tuple[int, int]:
        moves = self._get_candidate_moves(grid)
        if not moves:
            return BOARD_SIZE // 2, BOARD_SIZE // 2

        ordered_moves = self._order_moves(grid, moves, player)

        best_move = ordered_moves[0]
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for move in ordered_moves:
            r, c = move
            grid[r, c] = player
            score = self._minimax(grid, self.max_depth - 1, alpha, beta, False, player, last_move=(r, c))
            grid[r, c] = EMPTY

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move

    def _minimax(
        self,
        grid: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        ai_player: int,
        last_move: tuple[int, int] | None = None,
    ) -> float:
        game_over, winner = self._check_game_over(grid, last_move)
        if game_over:
            if winner == ai_player:
                return 1000000.0 + depth
            elif winner is not None:
                return -1000000.0 - depth
            else:
                return 0.0

        if depth == 0:
            return evaluate(grid, ai_player)

        moves = self._get_candidate_moves(grid)
        if not moves:
            return 0.0

        current_player = ai_player if maximizing else (O if ai_player == X else X)
        moves = self._order_moves(grid, moves, current_player)

        if maximizing:
            max_eval = -float("inf")
            for move in moves:
                r, c = move
                grid[r, c] = current_player
                eval_score = self._minimax(grid, depth - 1, alpha, beta, False, ai_player, last_move=(r, c))
                grid[r, c] = EMPTY
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                r, c = move
                grid[r, c] = current_player
                eval_score = self._minimax(grid, depth - 1, alpha, beta, True, ai_player, last_move=(r, c))
                grid[r, c] = EMPTY
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _get_candidate_moves(self, grid: np.ndarray) -> list[tuple[int, int]]:
        occupied = grid != EMPTY
        if not np.any(occupied):
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

        mask = np.zeros_like(occupied, dtype=bool)
        # Shift in 8 directions to find neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                
                # Manual shift logic for NumPy
                r_start, r_end = max(0, dr), min(BOARD_SIZE, BOARD_SIZE + dr)
                c_start, c_end = max(0, dc), min(BOARD_SIZE, BOARD_SIZE + dc)
                sr_start, sr_end = max(0, -dr), min(BOARD_SIZE, BOARD_SIZE - dr)
                sc_start, sc_end = max(0, -dc), min(BOARD_SIZE, BOARD_SIZE - dc)
                
                mask[r_start:r_end, c_start:c_end] |= occupied[sr_start:sr_end, sc_start:sc_end]

        candidates = np.argwhere((grid == EMPTY) & mask)
        return [tuple(m) for m in candidates]

    def _order_moves(
        self, grid: np.ndarray, moves: list[tuple[int, int]], player: int
    ) -> list[tuple[int, int]]:
        scored_moves = []
        for move in moves:
            r, c = move
            grid[r, c] = player
            try:
                score = evaluate(grid, player)
            finally:
                grid[r, c] = EMPTY
            scored_moves.append((score, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]

    def _check_game_over(self, grid: np.ndarray, last_move: tuple[int, int] | None = None) -> tuple[bool, int | None]:
        if last_move is not None:
            r, c = last_move
            player = grid[r, c]
            if player != EMPTY and is_win(grid, player, last_move):
                return True, player
            if is_draw(grid):
                return True, None
            return False, None
        
        # Fallback if no last_move
        if is_win(grid, X):
            return True, X
        if is_win(grid, O):
            return True, O
        if is_draw(grid):
            return True, None
        return False, None
