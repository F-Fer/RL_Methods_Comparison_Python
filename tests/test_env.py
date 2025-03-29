import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import GridEnv

class TestGridEnv(unittest.TestCase):

    def setUp(self):
        self.size = 4
        self.env = GridEnv(size=self.size)

    def test_reset_returns_valid_state(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, int)
        self.assertNotIn(self.env.state, self.env.terminal_states)

    def test_step_updates_state_correctly(self):
        self.env.state = (1, 1)
        obs, reward, done, truncated, info = self.env.step(0)  # up
        self.assertEqual(self.env.state, (1, 0))

        self.env.state = (1, 1)
        obs, reward, done, truncated, info = self.env.step(1)  # right
        self.assertEqual(self.env.state, (2, 1))

        self.env.state = (1, 1)
        obs, reward, done, truncated, info = self.env.step(2)  # down
        self.assertEqual(self.env.state, (1, 2))

        self.env.state = (1, 1)
        obs, reward, done, truncated, info = self.env.step(3)  # left
        self.assertEqual(self.env.state, (0, 1))

    def test_terminal_state_gives_reward_0_and_done(self):
        self.env.state = (0, 0)
        obs, reward, done, truncated, info = self.env.step(0)
        self.assertTrue(done)
        self.assertEqual(reward, 0)

    def test_invalid_action_raises_error(self):
        with self.assertRaises(ValueError):
            self.env.step(5)

    def test_set_state_invalid_input(self):
        with self.assertRaises(ValueError):
            self.env.set_state("invalid")

        with self.assertRaises(ValueError):
            self.env.set_state(999)

    def test_set_state_valid_non_terminal(self):
        state = self.env.get_state_from_coordinates((1, 1))
        result = self.env.set_state(state)
        self.assertTrue(result)
        self.assertEqual(self.env.state, (1, 1))

    def test_set_state_terminal_returns_false(self):
        state = self.env.get_state_from_coordinates((0, 0))
        result = self.env.set_state(state)
        self.assertFalse(result)

    def test_coordinate_conversion(self):
        for y in range(self.size):
            for x in range(self.size):
                state = self.env.get_state_from_coordinates((x, y))
                self.assertEqual(self.env.get_coordinates_from_state(state), (x, y))

if __name__ == "__main__":
    unittest.main()