import numpy as np
import random
import time
from src.ai.base import Agent
from src.ai.heuristic import evaluate
from src.game.constants import BOARD_SIZE, EMPTY, X, O
from src.game.rules import is_win, is_draw


class MinimaxAgent(Agent):
    def __init__(self, depth: int = 10, time_limit: float = 1.8) -> None:
        self.max_depth = depth
        self.time_limit = time_limit
        self.start_time = 0.0
        self.transposition_table = {}
        # Zobrist Hashing
        self.zobrist_table = np.random.randint(0, 2**63 - 1, (BOARD_SIZE, BOARD_SIZE, 3), dtype=np.int64)
        self.current_hash = 0

    def _get_hash(self, grid: np.ndarray) -> int:
        h = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                h ^= self.zobrist_table[r, c, grid[r, c]]
        return h

    def get_move(self, grid: np.ndarray, player: int) -> tuple[int, int]:
        self.start_time = time.time()
        self.current_hash = self._get_hash(grid)
        
        # Reset TT every move or keep it? Keeping it can help with transposition across moves, 
        # but let's clear it if memory is a concern. For 9x9, it should be fine to keep some.
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()

        moves = self._get_candidate_moves(grid)
        if not moves:
            return BOARD_SIZE // 2, BOARD_SIZE // 2

        best_move_overall = moves[0]
        
        # Iterative Deepening
        for depth in range(1, self.max_depth + 1):
            current_best_move = None
            max_eval = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            # Move ordering: prioritize best move from previous depth
            ordered_moves = self._order_moves(grid, moves, player, best_move_overall)
            
            try:
                for move in ordered_moves:
                    r, c = move
                    # Update hash and grid
                    old_val = grid[r, c]
                    self.current_hash ^= self.zobrist_table[r, c, old_val] ^ self.zobrist_table[r, c, player]
                    grid[r, c] = player
                    
                    score = self._minimax(grid, depth - 1, alpha, beta, False, player, last_move=(r, c))
                    
                    # Restore hash and grid
                    grid[r, c] = old_val
                    self.current_hash ^= self.zobrist_table[r, c, player] ^ self.zobrist_table[r, c, old_val]

                    if score > max_eval:
                        max_eval = score
                        current_best_move = move
                    alpha = max(alpha, max_eval)
                    
                    if time.time() - self.start_time > self.time_limit:
                        raise TimeoutError()
                
                if current_best_move:
                    best_move_overall = current_best_move
                
                # If we found a win, no need to search deeper
                if max_eval >= 500000:
                    break
                    
            except TimeoutError:
                break

        return best_move_overall

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
        # Check TT
        tt_entry = self.transposition_table.get(self.current_hash)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['score'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['score'])
            if alpha >= beta:
                return tt_entry['score']

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

        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()

        moves = self._get_candidate_moves(grid)
        if not moves:
            return 0.0

        current_player = ai_player if maximizing else (O if ai_player == X else X)
        # Move ordering: can use best move from TT if available
        best_tt_move = tt_entry['move'] if tt_entry else None
        moves = self._order_moves(grid, moves, current_player, best_tt_move)

        best_move = None
        if maximizing:
            max_eval = -float("inf")
            for move in moves:
                r, c = move
                old_val = grid[r, c]
                self.current_hash ^= self.zobrist_table[r, c, old_val] ^ self.zobrist_table[r, c, current_player]
                grid[r, c] = current_player
                
                eval_score = self._minimax(grid, depth - 1, alpha, beta, False, ai_player, last_move=(r, c))
                
                grid[r, c] = old_val
                self.current_hash ^= self.zobrist_table[r, c, current_player] ^ self.zobrist_table[r, c, old_val]
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            # Store in TT
            self._store_tt(self.current_hash, max_eval, depth, alpha, beta, best_move)
            return max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                r, c = move
                old_val = grid[r, c]
                self.current_hash ^= self.zobrist_table[r, c, old_val] ^ self.zobrist_table[r, c, current_player]
                grid[r, c] = current_player
                
                eval_score = self._minimax(grid, depth - 1, alpha, beta, True, ai_player, last_move=(r, c))
                
                grid[r, c] = old_val
                self.current_hash ^= self.zobrist_table[r, c, current_player] ^ self.zobrist_table[r, c, old_val]
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            # Store in TT
            self._store_tt(self.current_hash, min_eval, depth, alpha, beta, best_move)
            return min_eval

    def _store_tt(self, h, score, depth, alpha, beta, move):
        if score <= alpha:
            flag = 'UPPERBOUND'
        elif score >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        self.transposition_table[h] = {
            'score': score,
            'depth': depth,
            'flag': flag,
            'move': move
        }

    def _get_candidate_moves(self, grid: np.ndarray) -> list[tuple[int, int]]:
        occupied = grid != EMPTY
        if not np.any(occupied):
            return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

        mask = np.zeros_like(occupied, dtype=bool)
        # Shift in 8 directions to find neighbors up to distance 2
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0: continue
                
                # Using slices for faster neighbor detection
                r_start, r_end = max(0, dr), min(BOARD_SIZE, BOARD_SIZE + dr)
                c_start, c_end = max(0, dc), min(BOARD_SIZE, BOARD_SIZE + dc)
                sr_start, sr_end = max(0, -dr), min(BOARD_SIZE, BOARD_SIZE - dr)
                sc_start, sc_end = max(0, -dc), min(BOARD_SIZE, BOARD_SIZE - dc)
                
                mask[r_start:r_end, c_start:c_end] |= occupied[sr_start:sr_end, sc_start:sc_end]

        candidates = np.argwhere((grid == EMPTY) & mask)
        return [tuple(m) for m in candidates]

    def _order_moves(
        self, grid: np.ndarray, moves: list[tuple[int, int]], player: int, best_move_prev: tuple[int, int] | None = None
    ) -> list[tuple[int, int]]:
        scored_moves = []
        for move in moves:
            if move == best_move_prev:
                score = 10000000 # Ưu tiên tối đa nước đi tốt nhất từ depth trước
            else:
                r, c = move
                grid[r, c] = player
                try:
                    # Đánh giá sơ bộ nước đi dựa trên heuristic
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
        if is_win(grid, X): return True, X
        if is_win(grid, O): return True, O
        if is_draw(grid): return True, None
        return False, None
