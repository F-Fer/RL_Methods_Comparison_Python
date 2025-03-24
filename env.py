import gymnasium as gym
import numpy as np 

class GridEnv(gym.Env):
    """
    Simple grid environment of shape 4x4.
    The upper left and lower right are the terminal states.
    Rewards:
    - (-1) for all states, except terminal states
    - 0 for terminal states
    """

    def __init__(self):
        """
        Initialize the GridEnv environment.
        Sets up the observation space, action space, and initializes the state.
        """
        super().__init__()
        
        # Define observation space
        self.observation_space = gym.spaces.Discrete(4 * 4)

        # Define action space
        # Up, right, down, left
        self.action_space = gym.spaces.Discrete(4)

        # Define terminal states
        self.terminal_states = [(0, 0), (3, 3)]

        # Initialize at random position
        self.state = np.random.randint(0, 4), np.random.randint(0, 4)

    def reset(self, seed=None, options=None) -> (int, dict):
        """
        Reset the environment and return the initial observation.

        Parameters:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: The initial observation and an empty dictionary.
        """
        super().reset(seed=seed)

        # Initialize at random position
        self.state = np.random.randint(0, 4), np.random.randint(0, 4) 
        while self.state in self.terminal_states:
            self.state = np.random.randint(0, 4), np.random.randint(0, 4) 
        return self._get_obs(), {}

    def step(self, action: int) -> (int, float, bool, bool, {}):
        """
        Take a step in the direction of the specified action.

        Parameters:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left).

        Returns:
            tuple: A tuple containing the observation, reward, done flag, truncated flag, and an empty dictionary.
        """
        # Check type and range of action arg
        if not(type(action) == int):
            raise ValueError("Argument action must be of type int.")

        # Get current position and move according to action
        x, y = self.state
        match action:
            case 0:
                # Up
                y = min(y + 1, 3)

            case 1:
                x = min(x + 1, 3)

            case 2:
                y = max(y - 1, 0)

            case 3:
                x = max(x - 1, 0)

            case _:   
                raise ValueError("Argument action must be in range [0, 3]")
        
        # Set state
        self.state = x, y

        reward = -1
        done = False

        # Check if in terminal state
        if self.state in self.terminal_states:
            reward = 0
            done = True
        
        return self._get_obs(), reward, done, False, {}

    def set_starting_state(self, starting_state: (int, int)) -> None:
        """
        Set the starting state of the environment.

        Parameters:
            starting_state (tuple): A tuple representing the starting state (x, y).

        Raises:
            ValueError: If starting_state is not a tuple or if its elements are out of range.
        """
        if type(starting_state) is not tuple:
            raise ValueError("starting state must be a tuple.")

        x, y = starting_state

        if x == None or type(x) != int or x not in range(0, 4):
            raise ValueError("starting_states elements cannot be None, must be of type int, and must be in range [0, 3].")
        if y == None or type(y) != int or y not in range(0, 4):
            raise ValueError("starting_states elements cannot be None, must be of type int, and must be in range [0, 3].")

        self.state = x, y

    @staticmethod
    def get_coordinates_from_state(state):
        """
        Convert a state integer to its corresponding (x, y) coordinates.

        Parameters:
            state (int): The state integer.

        Returns:
            tuple: A tuple containing the (x, y) coordinates.
        """
        y = state % 4
        x = state // 4
        return x, y

    def _get_obs(self) -> int:
        """
        Map the current position of x and y coordinates onto a single integer representing a state.

        Returns:
            int: The mapped state integer in range [0, 15].
        """
        x, y = self.state
        return x * 4 + y 
        self.values = np.zeros((4, 4))