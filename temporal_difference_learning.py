import numpy as np
from env import GridEnv
from policy import Policy

class TemporalDifferenceLearning:
    """
    Temporal Difference Learning.
    """

    def __init__(self, env: GridEnv, policy: Policy, discount_factor: float = 0.9):
        self.env = env
        self.policy = policy
        self.values = np.zeros((4, 4))
        self.discount_factor = discount_factor
        # Initialize Q-values to small random values to break symmetry
        self.action_values = np.random.uniform(-0.1, 0.1, (16, 4))

    def evaluate_state_values (self, learning_rate: float = 0.1, num_episodes: int = 1000):
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.values[GridEnv.get_coordinates_from_state(state)] +=  learning_rate * (reward + self.discount_factor * self.values[GridEnv.get_coordinates_from_state(next_state)] - self.values[GridEnv.get_coordinates_from_state(state)])
                state = next_state

    def sarsa(self, learning_rate: float = 0.1, num_episodes: int = 1000):
        """State Action Reward State Action"""
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_action = self.policy(next_state)
                self.action_values[state, action] += learning_rate * (reward + self.discount_factor * self.action_values[next_state, next_action] - self.action_values[state, action])
                state = next_state
                action = next_action

    def q_learning(self, learning_rate: float = 0.1, num_episodes: int = 1000, epsylon: float = 0.1):
        """State Action Reward State Max Action"""
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                # Epsilon-greedy action selection
                if np.random.random() < epsylon:
                    action = int(np.random.choice(self.env.action_space.n))
                else:
                    action = int(np.argmax(self.action_values[state]))

                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Q-learning update
                if done:
                    target = reward
                else:
                    target = reward + self.discount_factor * np.max(self.action_values[next_state])
                
                self.action_values[state, action] += learning_rate * (target - self.action_values[state, action])
                state = next_state

    def improve_policy(self, epsylon: float = 0.1):
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

    def iterate(self, num_iterations: int = 10, episodes_per_eval: int = 1000, learning_rate: float = 0.1, epsylon: float = 0.1):
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