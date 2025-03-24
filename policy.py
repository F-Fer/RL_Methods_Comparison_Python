import numpy as np

class Policy:
    def __init__(self):
        """
        Initialize the Policy with uniform probabilities for each action.
        """
        self.probas = np.full((4, 4, 4), 0.25) # x, y, z, with z being the probability of taking the action ([0, 1])
        self.actions = np.array([0, 1, 2, 3]) # 0: up, 1: right ...

    def __call__(self, pos: (int, int)) -> int:
        """
        Get the action based on the current position using the policy.

        Parameters:
            pos (tuple): A tuple representing the position (x, y).

        Returns:
            int: The selected action.
        """
        x, y = pos
        probabilities = self.probas[x, y, :]

        return np.random.choice(self.actions, p=probabilities)

    def get_probability_of_action(self, pos: (int, int), action: int):
        """
        Get the probability of taking a specific action at a given position.

        Parameters:
            pos (tuple): A tuple representing the position (x, y).
            action (int): The action to evaluate.

        Returns:
            float: The probability of taking the specified action.
        """
        x, y = pos
        return self.probas[x, y, action]

    def display_policy(self):
        """
        Print the policy grid with the most probable action for each cell.
        Actions are represented as arrows:
        ↑ (up), → (right), ↓ (down), ← (left)
        Terminal states are shown as 'T'.
        """
        action_symbols = ["↑", "→", "↓", "←"]
        print("Policy:")
        for x in range(4):
            row = ""
            for y in range(4):
                probs = self.probas[x, y]
                if np.allclose(probs, [0.25, 0.25, 0.25, 0.25]):
                    row += " . "
                else:
                    best_action = np.argmax(probs)
                    row += f" {action_symbols[best_action]} "
            print(row)
