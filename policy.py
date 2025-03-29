import numpy as np

class Policy:
    def __init__(self, num_states: int):
        """
        Initialize the Policy with uniform probabilities for each action.
        
        Args:
            size (int): The size of the grid. The total number of states will be size * size.
        """
        self.num_states = num_states
        num_actions = 4
        self.probas = np.full((self.num_states, num_actions), 0.25)  # state × action probabilities
        self.actions = np.array([0, 1, 2, 3])  # 0: up, 1: right, 2: down, 3: left

    def __call__(self, state: int) -> int:
        """
        Get the action based on the current state using the policy.

        Args:
            state (int): The current state index.

        Returns:
            int: The selected action.
        """
        probabilities = self.probas[state, :]
        return int(np.random.choice(self.actions, p=probabilities))

    def get_probability_of_action(self, state: int, action: int) -> float:
        """
        Get the probability of taking a specific action at a given state.

        Args:
            state (int): The state index.
            action (int): The action to evaluate.

        Returns:
            float: The probability of taking the specified action.
        """
        return self.probas[state, action]
