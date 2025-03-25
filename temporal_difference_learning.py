import numpy as np
from env import GridEnv
from policy import Policy
from typing import Dict, Any
from rl.base import BaseRLAlgorithm

class TemporalDifferenceLearning(BaseRLAlgorithm):
    """
    Temporal Difference Learning implementation.
    
    This class implements various TD learning algorithms including:
    - TD(0) for state value estimation
    - SARSA for action-value estimation
    - Q-Learning for optimal action-value estimation
    
    The key difference between these algorithms is how they update their value estimates:
    - TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]
    - SARSA: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
    - Q-Learning: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, env: GridEnv, policy: Policy, discount_factor: float = 0.9):
        self.env = env
        self.policy = policy
        self.values = np.zeros((4, 4))
        self.discount_factor = discount_factor
        # Initialize Q-values to small random values to break symmetry
        self.action_values = np.random.uniform(-0.1, 0.1, (16, 4))

    def train(self, num_episodes: int = 1000, learning_rate: float = 0.1) -> Dict[str, Any]:
        """
        Train using Q-Learning (default) for optimal policy learning.
        
        Args:
            num_episodes: Number of episodes to train for
            learning_rate: Learning rate for updates
            
        Returns:
            Dictionary containing training metrics
        """
        return self.q_learning(learning_rate=learning_rate, num_episodes=num_episodes)

    def evaluate_state_values(self, learning_rate: float = 0.1, num_episodes: int = 1000):
        """
        TD(0) algorithm for state value estimation.
        
        Updates state values using the TD(0) update rule:
        V(s) ← V(s) + α[R + γV(s') - V(s)]
        """
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                # TD(0) update
                current_value = self.values[GridEnv.get_coordinates_from_state(state)]
                next_value = self.values[GridEnv.get_coordinates_from_state(next_state)]
                td_error = reward + self.discount_factor * next_value - current_value
                self.values[GridEnv.get_coordinates_from_state(state)] += learning_rate * td_error
                state = next_state

    def sarsa(self, learning_rate: float = 0.1, num_episodes: int = 1000):
        """
        SARSA algorithm for action-value estimation.
        
        Updates Q-values using the SARSA update rule:
        Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        """
        for _ in range(num_episodes):
            state, info = self.env.reset()
            action = self.policy(state)
            done = False
            while not done:
                next_state, reward, done, truncated, info = self.env.step(action)
                next_action = self.policy(next_state)
                # SARSA update
                current_q = self.action_values[state, action]
                next_q = self.action_values[next_state, next_action]
                td_error = reward + self.discount_factor * next_q - current_q
                self.action_values[state, action] += learning_rate * td_error
                state = next_state
                action = next_action

    def q_learning(self, learning_rate: float = 0.1, num_episodes: int = 1000, epsylon: float = 0.1):
        """
        Q-Learning algorithm for optimal action-value estimation.
        
        Updates Q-values using the Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
        
        Uses ε-greedy exploration strategy.
        """
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                # ε-greedy action selection
                if np.random.random() < epsylon:
                    action = int(np.random.choice(self.env.action_space.n))
                else:
                    action = int(np.argmax(self.action_values[state]))

                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Q-Learning update
                current_q = self.action_values[state, action]
                next_max_q = np.max(self.action_values[next_state]) if not done else 0
                td_error = reward + self.discount_factor * next_max_q - current_q
                self.action_values[state, action] += learning_rate * td_error
                state = next_state

    def improve_policy(self, epsylon: float = 0.1):
        """
        Improve the policy using ε-greedy exploration.
        
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

    def iterate(self, num_iterations: int = 10, episodes_per_eval: int = 1000, 
                learning_rate: float = 0.1, epsylon: float = 0.1):
        """
        Iterate between policy evaluation and improvement.
        
        Args:
            num_iterations: Number of policy evaluation-improvement cycles
            episodes_per_eval: Number of episodes per evaluation
            learning_rate: Learning rate for updates
            epsylon: Exploration rate for policy improvement
        """
        for _ in range(num_iterations):
            self.sarsa(learning_rate=learning_rate, num_episodes=episodes_per_eval)
            self.improve_policy(epsylon=epsylon)

    def generate_episode(self):
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