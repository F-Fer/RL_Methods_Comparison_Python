import numpy as np
from typing import Dict, Any
from env import GridEnv
from policy import Policy

class GPI:
    """
    Policy Iteration implementation following the classical algorithm.
    
    The algorithm consists of two main steps:
    1. Policy Evaluation: Compute V(s) for the current policy π
    2. Policy Improvement: Update π to be greedy with respect to V
    
    This process continues until the policy stabilizes (no changes in any state).
    """
    
    def __init__(self, env: GridEnv, discount_factor: float = 0.9):
        """
        Initialize Policy Iteration with a policy and value function.
        
        Args:
            env: The environment to learn in
            discount_factor: The discount factor (gamma) for future rewards
        """
        self.env = env
        self.policy = Policy(num_states=env.observation_space.n)  # π(s) ∈ A(s) arbitrarily for all s ∈ S
        self.values = np.zeros(env.observation_space.n)  # V(s) ∈ ℝ arbitrarily for all s ∈ S
        self.discount_factor = discount_factor

    def evaluate_policy(self, update_threshold: float = 0.001):
        """
        Evaluate the current policy by computing its value function.
        
        Implements the policy evaluation step using the Bellman equation:
        V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s',r|s,a)[r + γV_k(s')]
        
        In our deterministic environment, P(s',r|s,a) = 1 for the actual next state and reward,
        so we can simplify to:
        V_{k+1}(s) = Σ_a π(a|s)[r + γV_k(s')]
        
        Args:
            update_threshold: θ (small positive number determining accuracy)
        """
        while True:
            max_update = 0  # Δ ← 0
            for state in range(self.env.observation_space.n):
                old_value = self.values[state]  # v ← V(s)
                
                # Calculate new state value by summing over all actions
                new_value = 0
                for action in range(self.env.action_space.n):
                    # Get action probability from policy
                    action_prob = self.policy.get_probability_of_action(state, action)
                    
                    # Get next state and reward for this action

                    # MODEL-BASED: Uses the environment's transition model (via set_state + step)
                    is_valid = self.env.set_state(state)
                    if not is_valid:
                        continue
                    next_state, reward, done, truncated, info = self.env.step(action)
                    
                    # Add to value using action probability as weight
                    # BOOTSTRAPPING: Update V(s) using V(s') from previous iteration
                    new_value += action_prob * (reward + self.discount_factor * self.values[next_state])
                
                # Update value and track maximum change
                self.values[state] = new_value
                max_update = max(max_update, abs(old_value - new_value))  # Δ ← max(Δ, |v - V(s)|)
            
            if max_update < update_threshold:  # Δ < θ
                break

    def improve_policy(self) -> bool:
        """
        Improve the policy using value function.
        
        Implements the policy improvement step:
        policy_stable ← true
        For each s ∈ S:
            old_action ← π(s)
            π(s) ← argmax_a Σ p(s',r|s,a)[r + γV(s')]
            If old_action ≠ π(s), then policy_stable ← false
        
        Returns:
            bool: True if policy is stable (no changes), False otherwise
        """
        policy_stable = True  # policy-stable ← true
        
        for state in range(self.env.observation_space.n):
            # Skip terminal states
            x, y = self.env.get_coordinates_from_state(state)
            if (x, y) in self.env.terminal_states:
                continue
            
            # Store old action
            old_action = self.policy(state)  # old-action ← π(s)
            
            # Find best action using one-step lookahead
            best_action = None
            best_value = float("-inf")
            
            # π(s) ← argmax_a Σ p(s',r|s,a)[r + γV(s')]
            for action in range(self.env.action_space.n):
                self.env.set_state(state)
                obs, reward, done, truncated, info = self.env.step(action)
                value = reward + self.discount_factor * self.values[obs]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            # Update policy to choose best action deterministically
            for action in range(self.env.action_space.n):
                self.policy.probas[state, action] = 1.0 if action == best_action else 0.0
            
            # If old_action ≠ π(s), then policy_stable ← false
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable

    def iterate_policy(self, update_threshold: float = 0.001) -> Dict[str, Any]:
        """
        Run policy iteration until convergence.
        
        Implements the main policy iteration loop:
        1. Policy Evaluation
        2. Policy Improvement
        3. If policy is stable, return optimal value function and policy
        
        Returns:
            Dictionary containing the optimal value function and policy
        """
        while True:
            self.evaluate_policy(update_threshold)
            policy_stable = self.improve_policy()
            # If policy_stable, then stop and return V ≈ v* and π ≈ π*
            if policy_stable:
                break
        
        return {
            'values': self.values,
            'policy': self.policy
        }
