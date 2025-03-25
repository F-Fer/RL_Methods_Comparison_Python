import numpy as np
from env import GridEnv
class Policy:
    def __init__(self):
        """
        Initialize the Policy with uniform probabilities for each action.
        """
        self.probas = np.full((4, 4, 4), 0.25) # x, y, z, with z being the probability of taking the action ([0, 1])
        self.actions = np.array([0, 1, 2, 3]) # 0: up, 1: right ...

    def __call__(self, pos_or_state: (int, int) or int) -> int:
        """
        Get the action based on the current position or state using the policy.

        Parameters:
            pos_or_state (tuple or int): Either a (x, y) position tuple or a state index.

        Returns:
            int: The selected action.
        """
        if isinstance(pos_or_state, int):
            # Get coordinates from state    
            x, y = GridEnv.get_coordinates_from_state(pos_or_state)
        else:
            # Get coordinates from position
            x, y = pos_or_state

        probabilities = self.probas[x, y, :]
        return int(np.random.choice(self.actions, p=probabilities))

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
