import time
import gymnasium as gym
from policy_iteration import GPI
from monte_carlo import MonteCarlo
from policy import Policy
from utils import display_policy, display_values, display_action_values
from temporal_difference_learning import TemporalDifferenceLearning

def policy_iteration():
    gym.envs.registration.register(
        id="SimpleGrid-v0",
        entry_point="env:GridEnv",
    )

    env = gym.make("SimpleGrid-v0").unwrapped

    gpi = GPI()

    start_time = time.time()

    gpi.iterate_policy(env)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"\nTime taken: {elapsed_ms:.4f} ms")

    display_policy(gpi.policy)
    display_values(gpi.values)

def mc_estimate_state_values():
    gym.envs.registration.register(
        id="SimpleGrid-v0",
        entry_point="env:GridEnv",
    )

    env = gym.make("SimpleGrid-v0").unwrapped

    mc = MonteCarlo(env, Policy())

    for _ in range(10):
        mc.evaluate_state_values()

    display_values(mc.values)

def mc_estimate_action_values():
    gym.envs.registration.register(
        id="SimpleGrid-v0",
        entry_point="env:GridEnv",
    )

    env = gym.make("SimpleGrid-v0").unwrapped

    policy = Policy()

    mc = MonteCarlo(env, Policy())

    mc.iterate(100, 1000, 0.1)

    display_action_values(mc.action_values)
    display_policy(mc.policy)

def td_estimate_state_values():
    gym.envs.registration.register(
        id="SimpleGrid-v0",
        entry_point="env:GridEnv",
    )

    env = gym.make("SimpleGrid-v0").unwrapped

    policy = Policy()

    td = TemporalDifferenceLearning(env, policy)

    for _ in range(1000):
        td.evaluate_state_values()

    display_values(td.values)

if __name__ == "__main__":
    td_estimate_state_values() 