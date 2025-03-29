from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning algorithms."""
    
    # Environment settings
    grid_size: int = 4
    discount_factor: float = 0.9
    
    # Training settings
    learning_rate: float = 0.1
    num_episodes: int = 1000
    epsilon: float = 0.1
    
    # Policy iteration settings
    policy_iteration_iters: int = 10
    policy_eval_iters: int = 100
    
    # Monte Carlo settings
    mc_episodes: int = 100
    mc_iterations: int = 10
    mc_learning_rate: float = 0.1
    
    # TD Learning settings
    td_episodes: int = 1000
    td_iterations: int = 10
    td_learning_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'grid_size': self.grid_size,
            'discount_factor': self.discount_factor,
            'learning_rate': self.learning_rate,
            'num_episodes': self.num_episodes,
            'epsilon': self.epsilon,
            'policy_iteration_iters': self.policy_iteration_iters,
            'policy_eval_iters': self.policy_eval_iters,
            'mc_episodes': self.mc_episodes,
            'mc_learning_rate': self.mc_learning_rate,
            'td_episodes': self.td_episodes,
            'td_learning_rate': self.td_learning_rate
        }

# Default configuration
default_config = RLConfig() 