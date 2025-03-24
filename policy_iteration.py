import numpy as np
from env import GridEnv
from policy import Policy

class GPI:
    def __init__(self, discount_factor: float = 0.9):
        """
        Initialize the GPI with a policy and value function.

        Parameters:
            discount_factor (float, optional): The discount factor for future rewards.
        """
        self.policy = Policy()
        self.values = np.zeros((4, 4))
        self.discount_factor = discount_factor

    def evaluate_policy(self, env, update_threshold=0.01):
        """
        Parameters:
            env (GridEnv): The environment to evaluate the policy in.
            update_threshold (float, optional): The threshold for convergence.

        Returns:
            None
        """
        max_update = float("inf")
        while max_update > update_threshold:
            max_update = 0
            for state in range(env.observation_space.n):
                x, y = GridEnv.get_coordinates_from_state(state)
                if (x, y) in env.terminal_states:
                    continue
                state_value = 0

                for action in range(env.action_space.n):
                    env.set_starting_state((x, y))
                    action_proba = self.policy.get_probability_of_action((x, y), action)
                    
                    # Take step
                    obs, reward, done, truncated, info = env.step(action)
                    new_x, new_y = env.get_coordinates_from_state(obs)

                    # Calculate value with bellman equation
                    state_value += action_proba * (reward + self.discount_factor * self.values[new_x, new_y])

                # Calculate update size
                diff = abs(self.values[x, y] - state_value)
                max_update = max(max_update, diff)

                # Apply new value estimate
                self.values[x, y] = state_value

    def improve_policy(self, env, epsylon: float = 0.1):
        for state in range(env.observation_space.n):
            x, y = env.get_coordinates_from_state(state)
            if (x, y) in env.terminal_states:
                    continue
            best_action = 0
            highest_value = float("-inf")

            for action in range(env.action_space.n):
                env.set_starting_state((x, y))
                obs, reward, done, truncated, info = env.step(action)
                new_x, new_y = env.get_coordinates_from_state(obs)
                action_value = reward + self.discount_factor * self.values[new_x, new_y]
                if action_value > highest_value:
                    highest_value = action_value
                    best_action = action

            # Change policy
            for i in range(env.action_space.n):
                self.policy.probas[x, y, i] = 1 - epsylon + (epsylon / env.action_space.n) if i == best_action else epsylon / env.action_space.n

    def iterate_policy(self, env, max_iter=500_000, update_threshold: float = 0.9, epsylon: float = 0.1):
        for i in range(max_iter):
            old_policy = self.policy.probas.copy()
            self.evaluate_policy(env, update_threshold)
            self.improve_policy(env, epsylon)
            if np.array_equal(self.policy.probas, old_policy):
                print(f"Converged after {i} iteration.")
                break

    def display_values(self):
        print("Values:")
        for x in range(4):
            row = ""
            for y in range(4):
                row += f" {self.values[x, y]:.2f} "
            print(row)
