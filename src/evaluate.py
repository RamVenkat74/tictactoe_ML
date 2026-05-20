"""
Evaluate the trained agent and plot training curves.
=====================================================
Run:
    python src/evaluate.py
    python src/evaluate.py --episodes 5000 --model q_table.pkl
"""

import argparse
import random
from collections import defaultdict

from environment import TicTacToe, X, O
from q_agent import QLearningAgent


def evaluate(agent: QLearningAgent, n: int = 1000) -> dict:
    """
    Pit the greedy agent against a random opponent for n games.
    Returns win/loss/draw counts.
    """
    env = TicTacToe()
    results = defaultdict(int)

    for _ in range(n):
        env.reset()
        while not env.done:
            # Random X move
            x_moves = env.available_moves()
            if x_moves:
                env.step(random.choice(x_moves), player=X)
            if env.done:
                break

            # Greedy agent O move
            o_moves = env.available_moves()
            if o_moves:
                action = agent.choose_action(env.state_key(), o_moves, greedy=True)
                env.step(action, player=O)

        if env.winner == O:
            results["wins"] += 1
        elif env.winner == X:
            results["losses"] += 1
        else:
            results["draws"] += 1

    results["total"] = n
    return results


def plot_training_curve(episodes: int, alpha: float, gamma: float,
                        checkpoints: list, save_path: str):
    """
    Train agent incrementally and record win-rate at checkpoints.
    Saves a matplotlib plot to save_path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    from train import train as run_train

    win_rates = []
    loss_rates = []
    draw_rates = []

    agent = None
    prev = 0
    for cp in checkpoints:
        ep_count = cp - prev
        if ep_count <= 0:
            continue
        # train incrementally
        agent = QLearningAgent(alpha=alpha, gamma=gamma,
                               epsilon=agent.epsilon if agent else 1.0)
        from environment import TicTacToe, X, O
        import random as _random
        env = TicTacToe()
        for _ in range(ep_count):
            env.reset()
            while not env.done:
                xm = env.available_moves()
                if xm: env.step(_random.choice(xm), X)
                if env.done: break
                om = env.available_moves()
                if om:
                    st = env.state_key()
                    act = agent.choose_action(st, om)
                    r, done = env.step(act, O)
                    nst = env.state_key()
                    nm = env.available_moves() if not done else []
                    agent.update(st, act, r, nst, nm, done)
            agent.decay_epsilon()
        prev = cp
        res = evaluate(agent, n=500)
        win_rates.append(res["wins"] / 500 * 100)
        loss_rates.append(res["losses"] / 500 * 100)
        draw_rates.append(res["draws"] / 500 * 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(checkpoints[:len(win_rates)],  win_rates,  "g-o", label="Win %",  linewidth=2)
    ax.plot(checkpoints[:len(loss_rates)], loss_rates, "r-s", label="Loss %", linewidth=2)
    ax.plot(checkpoints[:len(draw_rates)], draw_rates, "b-^", label="Draw %", linewidth=2)
    ax.set_xlabel("Training episodes", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Q-Learning Agent Performance vs Training Episodes", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="q_table.pkl")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--plot",     action="store_true",
                        help="Generate training curve plot")
    parser.add_argument("--plot-out", default="training_curve.png")
    args = parser.parse_args()

    import os
    agent = QLearningAgent()
    if os.path.exists(args.model):
        agent.load(args.model)
    else:
        print(f"No model at {args.model} — train first with: python src/train.py")
        exit(1)

    print(f"\n  Evaluating agent over {args.episodes} games...\n")
    res = evaluate(agent, n=args.episodes)
    pct = lambda k: res[k] / res["total"] * 100
    print(f"  Win  : {res['wins']:>5,}  ({pct('wins'):5.1f}%)")
    print(f"  Loss : {res['losses']:>5,}  ({pct('losses'):5.1f}%)")
    print(f"  Draw : {res['draws']:>5,}  ({pct('draws'):5.1f}%)")
    print(f"  Total: {res['total']:>5,}\n")

    if args.plot:
        cps = [500, 1000, 2000, 5000, 10000, 20000, 50000]
        plot_training_curve(50000, 0.3, 0.9, cps, args.plot_out)
