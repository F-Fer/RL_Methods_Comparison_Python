# 🤖 Reinforcement Learning Methods Comparison

A Python implementation comparing different Reinforcement Learning (RL) algorithms in a grid world environment. This project implements and compares three fundamental RL approaches: Dynamic Programming (Policy Iteration), Monte Carlo Methods, and Temporal Difference Learning.

## 🎯 Features

- 🔲 Customizable Grid World environment
- 📊 Implementation of three core RL algorithms:
  - 🔄 Policy Iteration (Dynamic Programming)
  - 🎲 Monte Carlo Methods
  - ⏱️ Temporal Difference Learning (TD(0), SARSA, Q-Learning)
- 📈 Visualization utilities for policies and value functions
- ⚙️ Configurable hyperparameters
- 🧪 Comparative analysis framework

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL_Methods_Comparison_Python.git
cd RL_Methods_Comparison_Python
```

2. Install dependencies:
```bash
pip install numpy gymnasium
```

## 🚀 Usage

Run the main script to compare all algorithms:

```bash
python main.py
```

## 🏗️ Project Structure

- `env.py` - Grid World environment implementation
- `policy_iteration.py` - Dynamic Programming implementation
- `monte_carlo.py` - Monte Carlo methods implementation
- `temporal_difference_learning.py` - TD Learning algorithms
- `policy.py` - Policy class implementation
- `utils.py` - Visualization and helper functions
- `config.py` - Configuration parameters
- `main.py` - Main script to run comparisons

## 🎮 Environment

The environment is a customizable grid world where:
- States are numbered from 0 to (size²-1)
- Actions: Up (0), Right (1), Down (2), Left (3)
- Terminal states: Top-left and bottom-right corners
- Rewards: -1 for each move, 0 for reaching terminal states

## 🔧 Configuration

Key parameters can be modified in `config.py`:

```python
grid_size: int = 4              # Size of the grid
discount_factor: float = 0.9    # Gamma value
learning_rate: float = 0.1      # Alpha value
num_episodes: int = 1000        # Number of episodes
epsilon: float = 0.1            # Exploration rate
```

## 🧮 Implemented Algorithms

### Policy Iteration (Dynamic Programming)
- Model-based approach
- Alternates between policy evaluation and improvement
- Guaranteed convergence to optimal policy

### Monte Carlo Methods
- Model-free, episode-based learning
- First-visit MC for state/action value estimation
- Explores using ε-greedy policy

### Temporal Difference Learning
- Model-free, bootstrapping approach
- Implements TD(0), SARSA, and Q-Learning
- Combines MC and DP advantages

## 📊 Visualization

The project includes utilities to visualize:
- State value functions
- Action value functions (Q-values)
- Policies (using directional arrows)
- Grid world state

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- OpenAI Gym/Gymnasium documentation 