import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ai.rl_agent import AlphaZeroAgent, augment_data, encode_board
from src.game.board import Board
from src.game.constants import BOARD_SIZE, EMPTY, O, X
from src.game.rules import is_draw, is_win
from src.utils.logger import log_game_replay


TRAIN_LOG_FILE = os.path.join("logs", "training.csv")
TRAIN_LOG_FIELDS = [
    "episode",
    "loss",
    "buffer_size",
    "winner",
    "mcts_sims",
    "timestamp",
]


def play_self_play_game(
    agent: AlphaZeroAgent,
    exploration_moves: int = 12,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int | None, list[tuple[int, int, int]]]:
    board = Board()
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []
    moves: list[tuple[int, int, int]] = []
    current_player = X
    move_count = 0

    while True:
        state = encode_board(board.grid, current_player)
        temperature = 1.0 if move_count < exploration_moves else 0.0
        pi = agent.mcts.search(board.grid.copy(), current_player, temperature=temperature)

        valid_mask = (board.grid.flatten() == EMPTY).astype(np.float32)
        pi = pi * valid_mask
        total = float(pi.sum())
        if total <= 0:
            valid_indices = np.where(valid_mask > 0)[0]
            if len(valid_indices) == 0:
                winner = None
                break
            pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            pi[valid_indices] = 1.0 / len(valid_indices)
        else:
            pi = pi / total

        if temperature > 0:
            move_idx = int(np.random.choice(len(pi), p=pi))
        else:
            move_idx = int(np.argmax(pi))

        r, c = divmod(move_idx, BOARD_SIZE)
        board.place(r, c, current_player)

        states.append(state)
        policies.append(pi.astype(np.float32))
        players.append(current_player)
        moves.append((current_player, r, c))
        move_count += 1

        if is_win(board.grid, current_player, last_move=(r, c)):
            winner = current_player
            break
        if is_draw(board.grid):
            winner = None
            break

        current_player = O if current_player == X else X

    rewards: list[float] = []
    for player in players:
        if winner is None:
            rewards.append(0.0)
        elif winner == player:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    aug_states: list[np.ndarray] = []
    aug_policies: list[np.ndarray] = []
    aug_rewards: list[float] = []
    for state, policy, reward in zip(states, policies, rewards):
        for state_aug, policy_aug in augment_data(state, policy):
            aug_states.append(state_aug)
            aug_policies.append(policy_aug)
            aug_rewards.append(reward)

    return aug_states, aug_policies, aug_rewards, winner, moves


def _ensure_log_dir() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero agent via self-play")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of self-play episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--model-path", type=str, default="models/rl_agent.pth", help="Model save path")
    parser.add_argument("--replay-log", type=str, default="logs/replays.jsonl", help="Game replay log path")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (None=disabled)")
    parser.add_argument("--save-buffer", action="store_true", help="Save replay buffer periodically")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration coefficient")
    parser.add_argument("--exploration-moves", type=int, default=12, help="Opening moves sampled with temperature=1")
    parser.add_argument("--num-res-blocks", type=int, default=5, help="Number of residual blocks")
    parser.add_argument("--channels", type=int, default=64, help="ResNet channel width")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = AlphaZeroAgent(
        device=device,
        buffer_capacity=args.buffer_size,
        num_simulations=args.mcts_sims,
        c_puct=args.c_puct,
        num_res_blocks=args.num_res_blocks,
        channels=args.channels,
    )
    if os.path.exists(args.model_path):
        try:
            agent.load(args.model_path)
            print(f"Loaded existing model from {args.model_path}")
        except Exception:
            print("Could not load model, starting from scratch")

    _ensure_log_dir()
    file_exists = os.path.isfile(TRAIN_LOG_FILE)
    train_log = open(TRAIN_LOG_FILE, "a", newline="")
    writer = csv.DictWriter(train_log, fieldnames=TRAIN_LOG_FIELDS)
    if not file_exists:
        writer.writeheader()

    best_loss = float("inf")
    no_improve = 0

    for episode in range(1, args.episodes + 1):
        states, policies, rewards, winner, moves = play_self_play_game(
            agent,
            exploration_moves=args.exploration_moves,
        )
        agent.record_experience(states, policies, rewards)
        log_game_replay(args.replay_log, episode, moves, winner)

        winner_name = "X" if winner == X else ("O" if winner == O else "Draw")
        loss = 0.0
        if len(agent.buffer) >= args.batch_size:
            loss = agent.train_step(batch_size=args.batch_size)

            if args.patience is not None:
                if loss < best_loss:
                    best_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at episode {episode}")
                    break

        writer.writerow(
            {
                "episode": episode,
                "loss": f"{loss:.4f}",
                "buffer_size": len(agent.buffer),
                "winner": winner_name,
                "mcts_sims": args.mcts_sims,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if episode % 50 == 0 or episode == 1:
            print(
                f"Episode {episode:4d}/{args.episodes}, "
                f"Loss: {loss:.4f}, "
                f"Buffer: {len(agent.buffer)}, "
                f"Sims: {args.mcts_sims}"
            )

        if episode % args.save_every == 0:
            agent.save(args.model_path)
            if args.save_buffer:
                agent.save_buffer(args.model_path.replace(".pth", "_buffer.npz"))
            print("  -> Saved checkpoint")

    agent.save(args.model_path)
    if args.save_buffer:
        agent.save_buffer(args.model_path.replace(".pth", "_buffer.npz"))
    train_log.close()
    print(f"Training complete. Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
