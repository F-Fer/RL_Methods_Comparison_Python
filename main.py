import time
import gymnasium as gym
from typing import Dict, Any
from policy_iteration import GPI
from monte_carlo import MonteCarlo
from policy import Policy
from utils import display_policy, display_values, display_action_values, display_policy_from_action_values
from temporal_difference_learning import TemporalDifferenceLearning
from config import RLConfig, default_config

def setup_environment(config: RLConfig = default_config) -> gym.Env:
    """
    Set up and register the grid environment with configured size.
    
    Args:
        config: Configuration object containing grid_size and other parameters
        
    Returns:
        The created environment instance
    """
    try:
        gym.envs.registration.register(
            id="SimpleGrid-v0",
            entry_point="env:GridEnv",
            kwargs={'size': config.grid_size}
        )
    except gym.error.Error:
        # Environment already registered, unregister it first
        gym.envs.registration.registry.remove("SimpleGrid-v0")
        gym.envs.registration.register(
            id="SimpleGrid-v0",
            entry_point="env:GridEnv",
            kwargs={'size': config.grid_size}
        )
    
    return gym.make("SimpleGrid-v0", size=config.grid_size).unwrapped

def run_policy_iteration(config: RLConfig = default_config) -> Dict[str, Any]:
    """
    Run policy iteration algorithm.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Dictionary containing results and metrics
    """
    env = setup_environment(config)
    gpi = GPI(env, discount_factor=config.discount_factor)
    results = gpi.iterate_policy()
    return {
        'policy': results['policy'],
        'values': results['values']
    }

def run_monte_carlo(config: RLConfig = default_config) -> Dict[str, Any]:
    """
    Run Monte Carlo algorithm for both state and action value estimation.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Dictionary containing results and metrics
    """
    env = setup_environment(config)
    policy = Policy(env.size)  # Initialize policy with correct size
    mc = MonteCarlo(env, policy)
    
    # State value estimation
    for _ in range(config.mc_episodes):
        mc.evaluate_state_values()
    
    # Action value estimation
    mc.iterate(config.policy_iteration_iters, config.mc_episodes, config.mc_learning_rate)
    
    return {
        'state_values': mc.values,
        'action_values': mc.action_values,
        'policy': mc.policy
    }

def run_temporal_difference(config: RLConfig = default_config) -> Dict[str, Any]:
    """
    Run Temporal Difference Learning algorithms.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Dictionary containing results and metrics
    """
    env = setup_environment(config)
    policy = Policy(env.size)  # Initialize policy with correct size
    td = TemporalDifferenceLearning(env, policy)
    
    # State value estimation
    td.evaluate_state_values(config.td_learning_rate, config.td_episodes)
    
    # Action value estimation and Q-learning
    td.iterate(config.policy_iteration_iters, config.td_episodes, 
               config.td_learning_rate, config.epsilon)
    
    return {
        'state_values': td.values,
        'action_values': td.action_values,
        'policy': td.policy
    }

def display_results(results: Dict[str, Any], algorithm: str):
    """Display results for a given algorithm."""
    print(f"\n=== {algorithm} Results ===")
    
    if 'elapsed_time' in results:
        print(f"Time taken: {results['elapsed_time']:.4f} ms")
    
    if 'policy' in results:
        print("\nFinal Policy:")
        display_policy(results['policy'])
    
    if 'values' in results:
        print("\nState Values:")
        display_values(results['values'])
    
    if 'action_values' in results:
        print("\nAction Values:")
        display_action_values(results['action_values'])
        print("\nPolicy from Action Values:")
        display_policy_from_action_values(results['action_values'])

def main():
    """Run all RL algorithms and display results."""
    config = RLConfig()
    
    # Run and display results for each algorithm
    algorithms = {
        'Policy Iteration': run_policy_iteration,
        'Monte Carlo': run_monte_carlo,
        'Temporal Difference': run_temporal_difference
    }
    
    for name, algorithm in algorithms.items():
        try:
            results = algorithm(config)
            display_results(results, name)
        except Exception as e:
            print(f"Error running {name}: {str(e)}")

if __name__ == "__main__":
    main() 