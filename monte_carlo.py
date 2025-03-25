import numpy as np
from typing import Dict, Any, List, Tuple
from env import GridEnv
from policy import Policy
from rl.base import BaseRLAlgorithm

class MonteCarlo(BaseRLAlgorithm):
    """
    Monte Carlo Reinforcement Learning implementation.
    
    This class implements Monte Carlo methods for both state and action value estimation.
    The key features are:
    - First-visit Monte Carlo for state value estimation
    - First-visit Monte Carlo for action value estimation
    - Policy improvement using ε-greedy exploration
    
    The algorithm works by:
    1. Generating episodes using the current policy
    2. Computing returns for each state/action pair
    3. Averaging returns to estimate values
    4. Improving the policy based on the estimated values
    """

    def __init__(self, env: GridEnv, policy: Policy, discount_factor: float = 0.9):
        """
        Initialize Monte Carlo algorithm.
        
        Args:
            env: The environment to learn in
            policy: The policy to use for action selection
            discount_factor: The discount factor (gamma) for future rewards
        """
        super().__init__(env, policy, discount_factor)
        self.values = np.zeros((4, 4))
        self.action_values = np.zeros((16, 4))  # states x actions

    def train(self, num_episodes: int = 1000, learning_rate: float = 0.1) -> Dict[str, Any]:
        """
        Train using Monte Carlo methods.
        
        Args:
            num_episodes: Number of episodes to train for
            learning_rate: Not used in MC, kept for interface consistency
            
        Returns:
            Dictionary containing training metrics and results
        """
        # State value estimation
        self.evaluate_state_values(num_episodes)
        
        # Action value estimation and policy improvement
        self.iterate(10, num_episodes, 0.1)  # Default parameters
        
        return {
            'state_values': self.values,
            'action_values': self.action_values,
            'policy': self.policy
        }

    def evaluate_state_values(self, num_episodes: int = 1000):
        """
        First-visit Monte Carlo for state value estimation.
        
        For each episode:
        1. Generate episode using current policy
        2. For each state visited for the first time:
           - Compute return from that state
           - Add return to state's returns list
        3. Average returns to estimate state values
        
        Args:
            num_episodes: Number of episodes to evaluate
        """
        state_returns = {}
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited_states = set()
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.discount_factor * G
                
                if state not in visited_states:
                    visited_states.add(state)
                    if state not in state_returns:
                        state_returns[state] = []
                    state_returns[state].append(G)
        
        # Average returns to estimate state values
        for state in state_returns:
            self.values[GridEnv.get_coordinates_from_state(state)] = np.mean(state_returns[state])

    def evaluate_action_values(self, num_episodes: int = 1000):
        """
        First-visit Monte Carlo for action value estimation.
        
        For each episode:
        1. Generate episode using current policy
        2. For each state-action pair visited for the first time:
           - Compute return from that state-action
           - Add return to state-action's returns list
        3. Average returns to estimate action values
        
        Args:
            num_episodes: Number of episodes to evaluate
        """
        action_returns = {}
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            actions_taken = set()
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.discount_factor * G
                
                if (state, action) not in actions_taken:
                    actions_taken.add((state, action))
                    if (state, action) not in action_returns:
                        action_returns[(state, action)] = []
                    action_returns[(state, action)].append(G)
        
        # Average returns to estimate action values
        for (state, action) in action_returns:
            self.action_values[state, action] = np.mean(action_returns[(state, action)])

    def improve_policy(self, epsylon: float = 0.1):
        """
        Improve the policy using ε-greedy exploration.
        
        For each state:
        1. Find the action with highest estimated value
        2. Set that action's probability to 1 - ε + ε/|A|
        3. Set other actions' probabilities to ε/|A|
        
        Args:
            epsylon: Probability of choosing a random action
        """
        for state in range(self.env.observation_space.n):
            x, y = GridEnv.get_coordinates_from_state(state)
            if (x, y) in self.env.terminal_states:
                continue
                
            best_action = np.argmax(self.action_values[state])
            for action in range(self.env.action_space.n):
                if action == best_action:
                    self.policy.probas[x, y, action] = 1 - epsylon + (epsylon / self.env.action_space.n)
                else:
                    self.policy.probas[x, y, action] = epsylon / self.env.action_space.n

    def iterate(self, num_iterations: int = 10, episodes_per_eval: int = 1000, epsylon: float = 0.1):
        """
        Iterate between policy evaluation and improvement.
        
        Args:
            num_iterations: Number of policy evaluation-improvement cycles
            episodes_per_eval: Number of episodes per evaluation
            epsylon: Exploration rate for policy improvement
        """
        for _ in range(num_iterations):
            self.evaluate_action_values(num_episodes=episodes_per_eval)
            self.improve_policy(epsylon=epsylon)

    def generate_episode(self) -> list[tuple[int, int, float]]:
        obs, info = self.env.reset()
        episode = []
        done = False
        while not done:
            action = self.policy(obs)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode.append((obs, action, reward))
            obs = next_state
        return episode
    
if __name__ == "__main__":
    pass
