"""
Q-Learning Agent for Tic-Tac-Toe
==================================
Implements tabular Q-Learning with epsilon-greedy exploration.

Q-update rule:
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
"""

import random
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent that learns to play Tic-Tac-Toe.

    Attributes:
        alpha  (float): Learning rate (0, 1]
        gamma  (float): Discount factor [0, 1]
        epsilon(float): Exploration probability (decays over training)
        q_table(dict) : Maps (state, action) -> Q-value
    """

    def __init__(self, alpha=0.3, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: state_key -> list of Q-values (one per cell 0-8)
        self.q_table = defaultdict(lambda: [0.0] * 9)

    # ------------------------------------------------------------------
    # Core Q-learning helpers
    # ------------------------------------------------------------------

    def get_q(self, state: str, action: int) -> float:
        """Return Q-value for (state, action) pair."""
        return self.q_table[state][action]

    def update(self, state: str, action: int, reward: float,
               next_state: str, next_available: list, done: bool):
        """
        Apply the Bellman update:
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        """
        current_q = self.get_q(state, action)

        if done or not next_available:
            target = reward
        else:
            max_next_q = max(self.get_q(next_state, a) for a in next_available)
            target = reward + self.gamma * max_next_q

        self.q_table[state][action] += self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: str, available: list,
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If greedy=True, always pick the best known action (for evaluation).
        """
        if not available:
            raise ValueError("No available moves!")

        if not greedy and random.random() < self.epsilon:
            return random.choice(available)          # explore

        # exploit: pick action with highest Q-value
        q_vals = [(self.get_q(state, a), a) for a in available]
        return max(q_vals)[1]

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {path}  ({len(self.q_table)} states)")

    def load(self, path: str = "q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: [0.0] * 9, data)
        self.epsilon = self.epsilon_min   # switch to greedy mode
        print(f"Q-table loaded from {path}  ({len(self.q_table)} states)")

    @property
    def state_count(self) -> int:
        return len(self.q_table)
