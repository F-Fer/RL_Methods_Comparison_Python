import time
import gymnasium as gym
from policy_iteration import GPI

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

    gpi.policy.display_policy()
    gpi.display_values()

if __name__ == "__main__":
    policy_iteration()