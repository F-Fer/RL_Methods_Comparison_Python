import numpy as np
from policy import Policy

def display_values(values: np.ndarray):
    print("Values:")
    for x in range(4):
        row = ""
        for y in range(4):
            row += f" {values[x, y]:.2f} "
        print(row)

def display_policy(policy: Policy):
    """
    Print the policy grid with the most probable action for each cell.
    Actions are represented as arrows:
    ↑ (up), → (right), ↓ (down), ← (left)
    Terminal states are shown as 'T'.
    """
    action_symbols = ["→", "↓", "←", "↑"]
    print("Policy:")
    for x in range(4):
        row = ""
        for y in range(4):
            probs = policy.probas[x, y]
            if np.allclose(probs, [0.25, 0.25, 0.25, 0.25]):
                row += " . "
            else:
                best_action = np.argmax(probs)
                row += f" {action_symbols[best_action]} "
        print(row)

def display_policy_from_action_values(action_values: np.ndarray):
    action_symbols = ["→", "↓", "←", "↑"]
    print("Policy:")
    for x in range(4):
        row = ""
        for y in range(4):
            state = x * 4 + y
            probs = action_values[state]
            if np.allclose(probs, [0, 0, 0, 0]):
                row += " . "
            else:
                best_action = np.argmax(probs)
                row += f" {action_symbols[best_action]} "
        print(row)

def display_action_values(action_values: np.ndarray):
    print("Action Values:")
    print("State | Up    | Right | Down  | Left")
    print("-" * 40)
    for state in range(16):
        row = f"{state:5d} |"
        for action in range(4):
            row += f" {action_values[state, action]:5.2f} |"
        print(row)