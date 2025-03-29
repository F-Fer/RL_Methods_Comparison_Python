import numpy as np
from policy import Policy

def display_values(values: np.ndarray, grid_size: int = 4):
    """
    Display the values in a grid format.
    States are arranged as:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    
    Args:
        values: 1D array of state values
        grid_size: Size of the grid (default 4x4)
    """
    print("Values:")
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            state = row * grid_size + col
            row_str += f" {values[state]:6.2f}"
        print(row_str)

def display_policy(policy: Policy, grid_size: int = 4):
    """
    Print the policy grid with the most probable action for each cell.
    Actions are represented as arrows:
    0: ↑ (up), 1: → (right), 2: ↓ (down), 3: ← (left)
    Terminal states are shown as 'T'.
    
    States are arranged as:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    
    Args:
        policy: Policy object containing action probabilities
        grid_size: Size of the grid (default 4x4)
    """
    action_symbols = ["↑", "→", "↓", "←"]  # 0: up, 1: right, 2: down, 3: left
    print("Policy:")
    
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            state = row * grid_size + col
            probs = policy.probas[state]
            if np.allclose(probs, [0.25, 0.25, 0.25, 0.25]):
                row_str += "  .  "
            else:
                best_action = np.argmax(probs)
                row_str += f"  {action_symbols[best_action]}  "
        print(row_str)

def display_policy_from_action_values(action_values: np.ndarray, grid_size: int = 4):
    """
    Display a policy derived from action values in a grid format.
    Actions are represented as arrows:
    0: ↑ (up), 1: → (right), 2: ↓ (down), 3: ← (left)
    
    States are arranged as:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    
    Args:
        action_values: 2D array of shape (num_states, num_actions)
        grid_size: Size of the grid (default 4x4)
    """
    action_symbols = ["↑", "→", "↓", "←"]  # 0: up, 1: right, 2: down, 3: left
    print("Policy:")
    
    for row in range(grid_size):
        row_str = ""
        for col in range(grid_size):
            state = row * grid_size + col
            probs = action_values[state]
            if np.allclose(probs, [0, 0, 0, 0]):
                row_str += "  .  "
            else:
                best_action = np.argmax(probs)
                row_str += f"  {action_symbols[best_action]}  "
        print(row_str)

def display_action_values(action_values: np.ndarray):
    """
    Display action values for each state in a tabular format.
    Actions are: 0: up, 1: right, 2: down, 3: left
    
    Args:
        action_values: 2D array of shape (num_states, num_actions)
    """
    print("Action Values:")
    print("State |   Up   | Right  |  Down  |  Left")
    print("-" * 45)
    num_states = action_values.shape[0]
    for state in range(num_states):
        row = f"{state:5d} |"
        for action in range(4):
            row += f" {action_values[state, action]:6.2f} |"
        print(row)