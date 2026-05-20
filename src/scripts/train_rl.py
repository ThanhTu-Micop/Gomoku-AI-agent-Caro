import argparse
import csv
import os
import sys
from datetime import datetime
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.game.board import Board
from src.game.rules import is_win, is_draw
from src.game.constants import X, O, EMPTY, BOARD_SIZE
from src.ai.rl_agent import RLAgent, encode_board
from src.utils.logger import log_game_replay


TRAIN_LOG_FILE = os.path.join("logs", "training.csv")
TRAIN_LOG_FIELDS = ["episode", "loss", "buffer_size", "winner", "epsilon", "timestamp"]


def play_self_play_game(agent: RLAgent, epsilon: float = 0.1) -> tuple:
    board = Board()
    states = []
    policies = []
    moves = []
    current_player = X

    while True:
        state = encode_board(board.grid, current_player)
        states.append(state)

        # Epsilon-greedy exploration
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            winner = None
            break

        if np.random.random() < epsilon:
            # Random exploration
            idx = np.random.randint(len(valid_moves))
            r, c = valid_moves[idx]
        else:
            # Agent's policy
            r, c = agent.get_move(board.grid, current_player, deterministic=False)

        # Policy target (one-hot for the chosen move)
        policy_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        policy_target[r * BOARD_SIZE + c] = 1.0
        policies.append(policy_target)

        board.place(r, c, current_player)
        moves.append((current_player, r, c))

        if is_win(board.grid, current_player, last_move=(r, c)):
            winner = current_player
            break
        elif is_draw(board.grid):
            winner = None
            break

        current_player = O if current_player == X else X

    rewards = []
    for i in range(len(states)):
        turn = X if i % 2 == 0 else O
        if winner is None:
            rewards.append(0.0)
        elif winner == turn:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return states, policies, rewards, winner, moves


def _ensure_log_dir() -> None:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent via self-play")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of self-play episodes")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--model-path", type=str, default="models/rl_agent.pth", help="Model save path")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--epsilon-decay", action="store_true", help="Enable epsilon decay (0.5 -> 0.01)")
    parser.add_argument("--replay-log", type=str, default="logs/replays.jsonl", help="Game replay log path")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (None=disabled)")
    parser.add_argument("--save-buffer", action="store_true", help="Save replay buffer periodically")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    agent = RLAgent(device=device, buffer_capacity=args.buffer_size)
    if os.path.exists(args.model_path):
        try:
            agent.load(args.model_path)
            print(f"Loaded existing model from {args.model_path}")
        except:
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
        epsilon = args.epsilon
        if args.epsilon_decay:
            epsilon = max(0.01, 0.5 * (1 - episode / args.episodes))

        states, policies, rewards, winner, moves = play_self_play_game(agent, epsilon=epsilon)
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

        writer.writerow({
            "episode": episode,
            "loss": f"{loss:.4f}",
            "buffer_size": len(agent.buffer),
            "winner": winner_name,
            "epsilon": f"{epsilon:.4f}",
            "timestamp": datetime.now().isoformat(),
        })

        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode:4d}/{args.episodes}, Loss: {loss:.4f}, Buffer: {len(agent.buffer)}, Eps: {epsilon:.3f}")

        if episode % args.save_every == 0:
            agent.save(args.model_path)
            if args.save_buffer:
                agent.save_buffer(args.model_path.replace(".pth", "_buffer.npz"))
            print(f"  -> Saved checkpoint")

    agent.save(args.model_path)
    if args.save_buffer:
        agent.save_buffer(args.model_path.replace(".pth", "_buffer.npz"))
    train_log.close()
    print(f"Training complete. Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
