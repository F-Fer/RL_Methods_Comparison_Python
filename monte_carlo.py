import numpy as np
from typing import Dict, Any, List, Tuple
from env import GridEnv
from policy import Policy

class MonteCarlo:
    """
    Monte Carlo Methods (Model-Free, Sample-Based, No Bootstrapping)
    This method learns purely from episodes, without using the environment's internal model.
    
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

    def __init__(self, env: GridEnv, discount_factor: float = 0.9):
        """
        Initialize Monte Carlo algorithm.
        
        Args:
            env: The environment to learn in
            discount_factor: The discount factor (gamma) for future rewards
        """
        self.env = env
        self.policy = Policy(env.observation_space.n)
        self.discount_factor = discount_factor
        self.values = np.zeros(env.observation_space.n)  # State values
        self.action_values = np.zeros((env.observation_space.n, env.action_space.n))  # Q-values

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
            
            # SAMPLE-BASED: Waits until end of episode, computes actual return G
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
            # MODEL-FREE: Does not rely on knowledge of transition probabilities
            # Learns only from experience (sampled episodes), no set_state used
            self.values[state] = np.mean(state_returns[state])

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
            
            # SAMPLE-BASED: Waits until end of episode, computes actual return G
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
            # MODEL-FREE: Does not rely on knowledge of transition probabilities
            # Learns only from experience (sampled episodes), no set_state used
            self.action_values[state, action] = np.mean(action_returns[(state, action)])

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
              epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Iterate between policy evaluation and improvement (Monte Carlo Control).
        
        The algorithm alternates between:
        1. Policy Evaluation: Estimate Q(s,a) using Monte Carlo methods
        2. Policy Improvement: Make policy ε-greedy with respect to Q(s,a)
        
        Continues until either:
        - The policy is stable (no changes in action selection)
        - The maximum number of iterations is reached
        
        Note: Unlike in policy iteration, we maintain some exploration (ε > 0),
        so policy stability here means the best actions (argmax) remain the same.
        
        Args:
            max_iterations: Maximum number of policy evaluation-improvement cycles
            episodes_per_eval: Number of episodes per evaluation
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
            
            # Policy Evaluation: Estimate action values using Monte Carlo
            self.evaluate_action_values(num_episodes=episodes_per_eval)
            
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
