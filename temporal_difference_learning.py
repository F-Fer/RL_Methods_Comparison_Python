import numpy as np
from typing import Dict, Any, List, Tuple
from env import GridEnv
from policy import Policy

class TemporalDifferenceLearning:
    """
    Temporal Difference Learning implementation (Model-Free, Sample-Based, Bootstrapping).
    This method learns from individual transitions, using bootstrapping from value estimates.
    
    This class implements various TD learning algorithms including:
    - TD(0) for state value estimation
    - SARSA for action-value estimation
    - Q-Learning for optimal action-value estimation
    
    The key difference between these algorithms is how they update their value estimates:
    - TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]
    - SARSA: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
    - Q-Learning: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, env: GridEnv, discount_factor: float = 0.9):
        """
        Initialize TD Learning algorithm.
        
        Args:
            env: The environment to learn in
            discount_factor: The discount factor (gamma) for future rewards
        """
        self.env = env
        self.policy = Policy(env.observation_space.n)
        self.discount_factor = discount_factor
        self.values = np.zeros(env.observation_space.n)  # State values
        # Initialize Q-values to small random values to break symmetry
        self.action_values = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))

    def evaluate_state_values(self, learning_rate: float = 0.1, num_episodes: int = 1000):
        """
        TD(0) algorithm for state value estimation.
        
        Updates state values using the TD(0) update rule:
        V(s) ← V(s) + α[R + γV(s') - V(s)]
        
        Args:
            learning_rate: Step size for updates
            num_episodes: Number of episodes to evaluate
        """
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # TD(0) update
                # BOOTSTRAPPING: Uses current estimate of V(s') in the update
                current_value = self.values[state]
                next_value = self.values[next_state]
                td_error = reward + self.discount_factor * next_value - current_value
                self.values[state] += learning_rate * td_error
                
                state = next_state

    def sarsa(self, learning_rate: float = 0.1, num_episodes: int = 1000, epsilon: float = 0.1):
        """
        SARSA algorithm for action-value estimation.
        
        Updates Q-values using the SARSA update rule:
        Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        
        Args:
            learning_rate: Step size for updates
            num_episodes: Number of episodes to evaluate
            epsilon: Exploration rate for ε-greedy policy
        """
        for _ in range(num_episodes):
            state, info = self.env.reset()
            # Choose A from S using policy derived from Q (ε-greedy)
            action = self._epsilon_greedy_action(state, epsilon)
            done = False
            
            while not done:
                next_state, reward, done, truncated, info = self.env.step(action)
                next_action = self._epsilon_greedy_action(next_state, epsilon)
                
                # SARSA update (on-policy TD control)
                # BOOTSTRAPPING: Uses current estimate of Q(s',a') in the update
                current_q = self.action_values[state, action]
                next_q = self.action_values[next_state, next_action]
                td_error = reward + self.discount_factor * next_q - current_q
                self.action_values[state, action] += learning_rate * td_error
                
                state = next_state
                action = next_action

    def q_learning(self, learning_rate: float = 0.1, num_episodes: int = 1000, epsilon: float = 0.1):
        """
        Q-Learning algorithm for optimal action-value estimation.
        
        Updates Q-values using the Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            learning_rate: Step size for updates
            num_episodes: Number of episodes to evaluate
            epsilon: Exploration rate for ε-greedy policy
        """
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            
            while not done:
                # Choose A from S using policy derived from Q (ε-greedy)
                action = self._epsilon_greedy_action(state, epsilon)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Q-Learning update (off-policy TD control)
                # BOOTSTRAPPING: Uses current estimate of max_a' Q(s',a') in the update
                current_q = self.action_values[state, action]
                next_max_q = np.max(self.action_values[next_state]) if not done else 0
                td_error = reward + self.discount_factor * next_max_q - current_q
                self.action_values[state, action] += learning_rate * td_error
                
                state = next_state

    def improve_policy(self, epsilon: float = 0.1):
        """
        Improve the policy using ε-greedy exploration.
        
        For each state:
        1. Find the action with highest estimated value
        2. Set that action's probability to 1 - ε + ε/|A|
        3. Set other actions' probabilities to ε/|A|
        
        Args:
            epsilon: Probability of choosing a random action
        """
        for state in range(self.env.observation_space.n):
            # Skip terminal states
            is_valid = self.env.set_state(state)
            if not is_valid:
                continue
                
            best_action = np.argmax(self.action_values[state])
            for action in range(self.env.action_space.n):
                if action == best_action:
                    self.policy.probas[state, action] = 1 - epsilon + (epsilon / self.env.action_space.n)
                else:
                    self.policy.probas[state, action] = epsilon / self.env.action_space.n

    def iterate(self, max_iterations: int = 100, episodes_per_eval: int = 1000, 
              learning_rate: float = 0.1, epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Iterate between policy evaluation and improvement (TD Control).
        
        The algorithm alternates between:
        1. Policy Evaluation: Estimate Q(s,a) using SARSA or Q-learning
        2. Policy Improvement: Make policy ε-greedy with respect to Q(s,a)
        
        Continues until either:
        - The policy is stable (no changes in action selection)
        - The maximum number of iterations is reached
        
        Note: Unlike in policy iteration, we maintain some exploration (ε > 0),
        so policy stability here means the best actions (argmax) remain the same.
        
        Args:
            max_iterations: Maximum number of policy evaluation-improvement cycles
            episodes_per_eval: Number of episodes per evaluation
            learning_rate: Step size for updates
            epsilon: Exploration rate for policy improvement
            
        Returns:
            Dictionary containing:
            - 'converged': Whether the algorithm converged
            - 'iterations': Number of iterations performed
            - 'policy_stable': Whether the policy was stable in the last iteration
        """
        for iteration in range(max_iterations):
            # Store old best actions for each state
            old_best_actions = np.argmax(self.action_values, axis=1)
            
            # Policy Evaluation: Estimate action values using Q-learning
            self.q_learning(learning_rate=learning_rate, num_episodes=episodes_per_eval, epsilon=epsilon)
            
            # Policy Improvement: Update policy to be ε-greedy w.r.t. current Q-values
            self.improve_policy(epsilon=epsilon)
            
            # Check policy stability by comparing best actions
            new_best_actions = np.argmax(self.action_values, axis=1)
            policy_stable = np.all(old_best_actions == new_best_actions)
            
            if policy_stable:
                return {
                    'converged': True,
                    'iterations': iteration + 1,
                    'policy_stable': True
                }
        
        return {
            'converged': False,
            'iterations': max_iterations,
            'policy_stable': False
        }

    def _epsilon_greedy_action(self, state: int, epsilon: float) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return int(np.random.choice(self.env.action_space.n))
        return int(np.argmax(self.action_values[state]))

    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Generate a single episode using the current policy.
        
        Returns:
            List of (state, action, reward) tuples representing the episode
        """
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