import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PathTrackingEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, height=1.0, width=1.0, noise_std=0):
        """Initialise the path tracking environment.
            :param height: Height of the world.
            :param width: Width of the world
            :param noise_std: Standard deviation of Gaussian noise added to action, in degrees.
            """
        self.height = height
        self.width = width
        self.noise_std = noise_std
        self.step_size = 0.1
        self.start_state = (0.5, 0.0)               # y, x
        self.goal_box = (0.4, 0.9, 0.6, 1.0)        # y1, x1, y2, x2
        # TODO implement lava boxes

        self.action_space = spaces.Box(0, 360, shape=(1,))
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.height, self.width]))

        self._seed()
        self.viewer = None
        self.state = None

        self.RAD_TO_DEG = 57.2957795

    def in_goal_box(self, y, x):
        y1, x1, y2, x2 = self.goal_box
        if y1 <= y and y <= y2 and x1 <= x and x <= x2:
            return True
        return False


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        y, x = state  # y is vertical coordinate, x is horizontal
        if self.noise_std > 0:
            action += np.random.normal(0, self.noise_std)

        angle = float(action) / self.RAD_TO_DEG
        distance = self.step_size
        dy = distance * math.sin(angle)
        dx = distance * math.cos(angle)

        new_y = min(self.height, max(0, y + dy))  # Make sure we're within bounds
        new_x = min(self.width, max(0, x + dx))

        self.state = new_y, new_x

        reward = 0.0
        done = False
        if self.in_goal_box(new_y, new_x):
            reward = 1.0
            done = True

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.start_state
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        pass  # TODO
