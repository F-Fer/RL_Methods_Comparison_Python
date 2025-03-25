import numpy as np
from typing import Dict, Any
from env import GridEnv
from policy import Policy
from rl.base import BaseRLAlgorithm

class GPI(BaseRLAlgorithm):
    """
    Generalized Policy Iteration (GPI) implementation.
    
    GPI alternates between two steps:
    1. Policy Evaluation: Compute the value function for the current policy
    2. Policy Improvement: Update the policy based on the value function
    
    The algorithm converges to the optimal policy when the policy stops changing.
    """
    
    def __init__(self, discount_factor: float = 0.9):
        """
        Initialize GPI with a policy and value function.
        
        Args:
            discount_factor: The discount factor (gamma) for future rewards
        """
        super().__init__(None, Policy(), discount_factor)  # env will be set in train()
        self.values = np.zeros((4, 4))

    def train(self, num_episodes: int = 1000, learning_rate: float = 0.1) -> Dict[str, Any]:
        """
        Train the policy using GPI.
        
        Args:
            num_episodes: Maximum number of policy iterations
            learning_rate: Not used in GPI, kept for interface consistency
            
        Returns:
            Dictionary containing training metrics and results
        """
        if self.env is None:
            raise ValueError("Environment not set. Call set_environment() first.")
            
        start_time = time.time()
        self.iterate_policy(self.env)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'elapsed_time': elapsed_ms,
            'policy': self.policy,
            'values': self.values
        }

    def evaluate_policy(self, env: GridEnv, update_threshold: float = 0.01):
        """
        Evaluate the current policy by computing its value function.
        
        Uses the Bellman equation for policy evaluation:
        V(s) = Σ π(a|s) * Σ p(s',r|s,a) * [r + γV(s')]
        
        Args:
            env: The environment to evaluate the policy in
            update_threshold: The threshold for convergence
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
                    
                    # Take step and get next state
                    obs, reward, done, truncated, info = env.step(action)
                    new_x, new_y = GridEnv.get_coordinates_from_state(obs)
                    
                    # Bellman equation update
                    state_value += action_proba * (reward + self.discount_factor * self.values[new_x, new_y])
                
                # Track maximum update for convergence
                diff = abs(self.values[x, y] - state_value)
                max_update = max(max_update, diff)
                
                # Update value estimate
                self.values[x, y] = state_value

    def improve_policy(self, env: GridEnv, epsylon: float = 0.1):
        """
        Improve the policy using the current value function.
        
        Uses ε-greedy policy improvement:
        π(a|s) = 1 - ε + ε/|A| if a is optimal
        π(a|s) = ε/|A| otherwise
        
        Args:
            env: The environment to improve the policy in
            epsylon: Probability of choosing a random action
        """
        for state in range(env.observation_space.n):
            x, y = GridEnv.get_coordinates_from_state(state)
            if (x, y) in env.terminal_states:
                continue
                
            # Find best action
            best_action = 0
            highest_value = float("-inf")
            
            for action in range(env.action_space.n):
                env.set_starting_state((x, y))
                obs, reward, done, truncated, info = env.step(action)
                new_x, new_y = GridEnv.get_coordinates_from_state(obs)
                action_value = reward + self.discount_factor * self.values[new_x, new_y]
                
                if action_value > highest_value:
                    highest_value = action_value
                    best_action = action
            
            # Update policy probabilities
            for action in range(env.action_space.n):
                if action == best_action:
                    self.policy.probas[x, y, action] = 1 - epsylon + (epsylon / env.action_space.n)
                else:
                    self.policy.probas[x, y, action] = epsylon / env.action_space.n

    def iterate_policy(self, env: GridEnv, max_iter: int = 500_000, 
                      update_threshold: float = 0.9, epsylon: float = 0.1):
        """
        Run policy iteration until convergence or max iterations reached.
        
        Args:
            env: The environment to learn in
            max_iter: Maximum number of iterations
            update_threshold: Threshold for policy evaluation convergence
            epsylon: Exploration rate for policy improvement
        """
        for i in range(max_iter):
            old_policy = self.policy.probas.copy()
            self.evaluate_policy(env, update_threshold)
            self.improve_policy(env, epsylon)
            
            # Check for convergence
            if np.array_equal(self.policy.probas, old_policy):
                break
