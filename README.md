# Snake Game with Deep Q-Learning

This project implements a classic Snake game where the snake learns to play using Deep Q-Learning (DQN), a reinforcement learning technique.

## Requirements

- Rust (install via [rustup](https://rustup.rs/))
- Gnuplot (for visualization):
  ```bash
  sudo apt install gnuplot
  ```

## How It Works

The snake learns by:
1. Observing its environment (position, direction, food location, etc.)
2. Making decisions (move straight, left, or right)
3. Receiving rewards (+10 for food, -10 for collision)
4. Adjusting its neural network based on experience

Key components:
- `LinearQNet`: A simple neural network (2 linear layers)
- `QTrainer`: Handles the DQN training process
- `Agent`: Manages exploration vs exploitation and memory
- `SnakeGame`: The game environment

## Running

```bash
cargo run
```

The game will display:
- Blue snake with dark blue segments
- Red food
- Current score in the top-left
- Performance plot in a separate gnuplot window

## Training

The agent starts with random exploration (ε=80%) and gradually relies more on its learned model as ε decreases. The plot shows score progression across games.
