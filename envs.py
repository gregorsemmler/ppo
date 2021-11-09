import numpy as np
from gym import Env, spaces


class SimpleCorridorEnv(Env):

    def __init__(self, half_width=5, equal_reward=True):
        self.state = None
        self.half_width = 5
        self.length = 2 * half_width + 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, dtype=np.uint8, shape=(self.length,))
        self.equal_reward = equal_reward

    def get_reward_for_state(self, state):
        if self.equal_reward:
            return -1.0
        curr_pos = state.argmax()
        reward = float(self.length - curr_pos)
        return reward

    def step(self, action):
        action = int(action)
        if action < 0 or action > 1:
            raise ValueError(f"Invalid action {action}")

        curr_pos = self.state.argmax()
        self.state = np.zeros((self.length, ), dtype=np.uint8)
        if action == 0:
            new_pos = max(0, curr_pos-1)
        else:
            new_pos = min(self.length-1, curr_pos+1)

        self.state[new_pos] = 1
        reward = self.get_reward_for_state(self.state)
        done = new_pos == self.length - 1
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros((self.length, ), dtype=np.uint8)
        self.state[self.half_width] = 1
        return self.state

    def render(self, mode='human'):
        return np.zeros((1, 1, 3), dtype=np.uint8)
