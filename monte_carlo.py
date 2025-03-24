import numpy as np
from env import GridEnv
from policy import Policy

class MonteCarlo:
    """
    Monte Carlo first visit policy evaluation.
    """

    def __init__(self, env: GridEnv, policy: Policy, discount_factor: float = 0.9):
        self.env = env
        self.policy = policy
        self.values = np.zeros((4, 4))
        self.discount_factor = discount_factor

    def evaluate_policy(self, num_episodes: int = 1000):
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
            
        for state in state_returns:
            self.values[GridEnv.get_coordinates_from_state(state)] = np.mean(state_returns[state])

    def iterate(self, num_iterations: int = 10, episodes_per_eval: int = 1000):
        for _ in range(num_iterations):
            self.evaluate_policy(num_episodes=episodes_per_eval)

    def generate_episode(self) -> list[tuple[int, int, float]]:
        episode = []
        obs, info = self.env.reset()
        done = False
        while not done:
            action = self.policy.call_from_state(obs)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode.append((obs, action, reward))
            obs = next_state
        return episode
    
    def display_values(self):
        print("Values:")
        for x in range(4):
            row = ""
            for y in range(4):
                row += f" {self.values[x, y]:.2f} "
            print(row)
