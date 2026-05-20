# Tic-Tac-Toe — Q-Learning Agent

Reinforcement Learning assignment: a Q-Learning agent that learns to play
Tic-Tac-Toe through self-play against a random opponent.

---

## Project structure

```
tictactoe-ql/
├── src/
│   ├── environment.py   # Game logic (board, step, reward)
│   ├── q_agent.py       # Q-Learning agent (Q-table, update rule, ε-greedy)
│   ├── train.py         # Training loop (self-play)
│   ├── play.py          # Terminal UI — play vs trained agent
│   └── evaluate.py      # Win-rate evaluation + optional plot
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent
```bash
cd src
python train.py
# Optional flags:
python train.py --episodes 50000 --alpha 0.3 --gamma 0.9
```

### 3. Play against it
```bash
python play.py
# Train then play in one step:
python play.py --train-first
```

### 4. Evaluate win rate
```bash
python evaluate.py --episodes 2000
python evaluate.py --episodes 2000 --plot   # saves training_curve.png
```

---

## Algorithm: Q-Learning

Q-Learning is a **model-free, off-policy** temporal difference (TD) algorithm.

### Bellman update rule

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

| Symbol | Name | Description |
|--------|------|-------------|
| `s` | state | board configuration (string key) |
| `a` | action | cell index 0–8 |
| `α` | learning rate | how fast new info overwrites old (0.1–0.5) |
| `γ` | discount factor | weight of future rewards (0.9 typical) |
| `r` | reward | +1 win, −1 loss, +0.5 draw, 0 otherwise |
| `ε` | epsilon | exploration probability (decays 1.0 → 0.05) |

### Epsilon-greedy exploration

- With probability **ε** → pick a **random** move (explore)
- Otherwise → pick the **best known** move from Q-table (exploit)
- ε decays each episode so the agent explores early and exploits later

### State representation

Each board is encoded as a 9-character string, e.g. `"102010002"`.
This becomes the key in the Q-table dictionary.

### Rewards

| Outcome | Reward |
|---------|--------|
| Agent wins | +1.0 |
| Agent loses | −1.0 |
| Draw | +0.5 |
| Intermediate step | 0.0 |

---

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| episodes | 20 000 | More = better learning |
| α (alpha) | 0.3 | Higher = learns faster, less stable |
| γ (gamma) | 0.9 | Higher = values long-term rewards more |
| ε start | 1.0 | Full exploration at start |
| ε end | 0.05 | ~5% random moves at end |
| ε decay | 0.9995 | Exponential decay per episode |

---

## Requirements

- Python 3.8+
- `matplotlib` (optional, for evaluation plots)
