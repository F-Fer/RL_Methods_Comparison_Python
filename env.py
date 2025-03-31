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

    def __init__(self, size: int):
        """
        Initialize the GridEnv environment.
        Sets up the observation space, action space, and initializes the state.
        """
        super().__init__()
        self.size = size
        # Define observation space
        self.observation_space = gym.spaces.Discrete(size * size)

        # Define action space
        # Up, right, down, left
        self.action_space = gym.spaces.Discrete(4)

        # Define terminal states
        self.terminal_states = [(0, 0), (size - 1, size - 1)]

        # Initialize at random position
        self.state = np.random.randint(0, size), np.random.randint(0, size)

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
        self.state = np.random.randint(0, self.size), np.random.randint(0, self.size) 
        while self.state in self.terminal_states:
            self.state = np.random.randint(0, self.size), np.random.randint(0, self.size) 
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
                y = max(y - 1, 0)

            case 1:
                # Right
                x = min(x + 1, self.size - 1)

            case 2:
                # Down
                y = min(y + 1, self.size - 1)

            case 3:
                # Left
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

    def set_state(self, starting_state: int) -> bool:
        """
        Set the starting state of the environment.

        Parameters:
            starting_state (int): An integer representing the starting state.

        Raises:
            ValueError: If starting_state is not an integer or if its elements are out of range.

        Returns:
            bool: True if the state was set successfully and is valid, False otherwise.
        """
        if type(starting_state) is not int:
            raise ValueError("state must be an integer.")

        x, y = self.get_coordinates_from_state(starting_state)

        if x == None or type(x) != int or x not in range(0, self.size):
            raise ValueError("starting_states elements cannot be None, must be of type int, and must be in range [0, 3].")
        if y == None or type(y) != int or y not in range(0, self.size):
            raise ValueError("starting_states elements cannot be None, must be of type int, and must be in range [0, 3].")
        
        if (x, y) in self.terminal_states:  
            return False

        self.state = x, y
        return True

    def get_coordinates_from_state(self, state : int) -> (int, int):
        """
        Convert a state integer to its corresponding (x, y) coordinates.

        Parameters:
            state (int): The state integer.

        Returns:
            tuple: A tuple containing the (x, y) coordinates.
        """
        x = state % self.size
        y = state // self.size
        return x, y

    def get_state_from_coordinates(self, coordinates: (int, int)) -> int:
        x, y = coordinates
        return y * self.size + x

    def _get_obs(self) -> int:
        """
        Map the current position of x and y coordinates onto a single integer representing a state.

        Returns:
            int: The mapped state integer in range [0, 15].
        """
        x, y = self.state
        return y * self.size + x 
    
    def display_state(self):
        """
        Display the current state as a grid, showing the agent's position with 'A',
        terminal states with 'T', and empty cells with '.'.

        Example output:
        . . . .
        . A . T
        . . . .
        T . . T
        """
        grid = [[f'.' for col in range(self.size)] for row in range(self.size)]
        
        # Mark terminal states
        for tx, ty in self.terminal_states:
            grid[ty][tx] = 'T'
            
        # Mark agent position
        x, y = self.state
        grid[y][x] = 'A'
        
        # Print grid
        for row in grid:
            print(' '.join(row))
    

if __name__ == "__main__":
    env = GridEnv(size=4)

    print(env.reset())
    env.display_state()
    print(env.state)
    print()

    print(env.step(0))
    env.display_state()
    print(env.state)
    print()

    print(env.step(1))
    env.display_state()
    print(env.state)    
    print()

    print(env.step(2))  
    env.display_state()
    print(env.state)
    print()

    print(env.step(3))
    env.display_state()