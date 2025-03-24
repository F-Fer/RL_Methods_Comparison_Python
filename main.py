import time
import gymnasium as gym
from policy_iteration import GPI
from monte_carlo import MonteCarlo
from policy import Policy
from utils import display_policy, display_values

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

def monte_carlo_policy_evaluation():
    gym.envs.registration.register(
        id="SimpleGrid-v0",
        entry_point="env:GridEnv",
    )

    env = gym.make("SimpleGrid-v0").unwrapped

    mc = MonteCarlo(env, Policy())

    mc.iterate()

    display_values(mc.values)

if __name__ == "__main__":
    monte_carlo_policy_evaluation()