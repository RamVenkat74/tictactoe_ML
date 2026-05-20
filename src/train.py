"""
Training Loop
=============
Trains the Q-Learning agent by self-play against a random opponent.

Run:
    python src/train.py
    python src/train.py --episodes 50000 --alpha 0.3 --gamma 0.9
"""

import argparse
import random
import time
from collections import deque

from environment import TicTacToe, X, O
from q_agent import QLearningAgent


def random_move(available: list) -> int:
    """Simple random opponent (plays as X)."""
    return random.choice(available)


def train(episodes: int = 20_000, alpha: float = 0.3,
          gamma: float = 0.9, save_path: str = "q_table.pkl",
          print_every: int = 2000) -> QLearningAgent:
    """
    Train the Q-learning agent for `episodes` self-play games.

    Training protocol
    -----------------
    - Agent plays as O (player 2); random opponent plays as X (player 1).
    - X always moves first.
    - Epsilon decays from 1.0 → 0.05 over the course of training.
    - After each agent move, the Bellman update is applied.

    Returns the trained agent.
    """
    agent = QLearningAgent(alpha=alpha, gamma=gamma)
    env = TicTacToe()

    wins = losses = draws = 0
    recent = deque(maxlen=1000)   # sliding window for live win-rate

    print(f"\n{'='*50}")
    print(f"  Training Q-Learning agent")
    print(f"  Episodes : {episodes:,}")
    print(f"  Alpha (α): {alpha}   Gamma (γ): {gamma}")
    print(f"{'='*50}\n")

    t0 = time.time()

    for ep in range(1, episodes + 1):
        env.reset()
        prev_state = None
        prev_action = None

        while not env.done:
            # --- X moves (random opponent) ---
            x_moves = env.available_moves()
            if not x_moves:
                break
            x_action = random_move(x_moves)
            reward, done = env.step(x_action, player=X)

            if done:
                # Agent lost or draw after X's move
                if prev_state is not None:
                    agent.update(prev_state, prev_action, reward,
                                 env.state_key(), [], done=True)
                break

            # --- Agent (O) moves ---
            state = env.state_key()
            o_moves = env.available_moves()
            if not o_moves:
                break

            action = agent.choose_action(state, o_moves)
            reward, done = env.step(action, player=O)

            next_state = env.state_key()
            next_moves = env.available_moves() if not done else []

            agent.update(state, action, reward, next_state, next_moves, done)

            prev_state = next_state
            prev_action = action

            if done:
                break

        # --- Track results ---
        if env.winner == O:
            wins += 1; recent.append("W")
        elif env.winner == X:
            losses += 1; recent.append("L")
        else:
            draws += 1; recent.append("D")

        agent.decay_epsilon()

        # --- Progress report ---
        if ep % print_every == 0:
            total = wins + losses + draws
            wr = wins / total * 100
            lr = losses / total * 100
            dr = draws / total * 100
            recent_wr = recent.count("W") / len(recent) * 100
            elapsed = time.time() - t0
            print(f"Ep {ep:>7,} / {episodes:,}  |  "
                  f"W {wr:5.1f}%  L {lr:5.1f}%  D {dr:5.1f}%  |  "
                  f"Recent-1k W: {recent_wr:5.1f}%  |  "
                  f"ε={agent.epsilon:.3f}  "
                  f"States={agent.state_count:,}  "
                  f"[{elapsed:.1f}s]")

    # --- Final summary ---
    total = wins + losses + draws
    print(f"\n{'='*50}")
    print(f"  Training complete in {time.time()-t0:.1f}s")
    print(f"  Agent wins  : {wins:,}  ({wins/total*100:.1f}%)")
    print(f"  Agent losses: {losses:,}  ({losses/total*100:.1f}%)")
    print(f"  Draws       : {draws:,}  ({draws/total*100:.1f}%)")
    print(f"  Q-table size: {agent.state_count:,} states")
    print(f"  Final ε     : {agent.epsilon:.4f}")
    print(f"{'='*50}\n")

    agent.save(save_path)
    return agent


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning Tic-Tac-Toe agent")
    parser.add_argument("--episodes",  type=int,   default=20_000)
    parser.add_argument("--alpha",     type=float, default=0.3,  help="Learning rate")
    parser.add_argument("--gamma",     type=float, default=0.9,  help="Discount factor")
    parser.add_argument("--save",      type=str,   default="q_table.pkl")
    parser.add_argument("--print-every", type=int, default=2000)
    args = parser.parse_args()

    train(episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
          save_path=args.save, print_every=args.print_every)
