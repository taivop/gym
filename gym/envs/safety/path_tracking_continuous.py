import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PathTrackingEnv(gym.Env):
    metadata = {
        'render.modes': ['rgb_array', 'human']
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

        self.action_space = spaces.Box(0, 360, shape=(1,))
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.height, self.width]))

        self._seed()
        self.figure = None
        self.ax = None
        self.state = None

        self.RAD_TO_DEG = 57.2957795

    def in_goal_box(self, y, x):
        return self.in_box(y, x, self.goal_box)

    @staticmethod
    def in_box(y, x, box):
        y1, x1, y2, x2 = box
        assert y1 < y2, "box y-dimensions incoherent: y1 must be less than y2"
        assert x1 < x2, "box x-dimensions incoherent: x1 must be less than x2"
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
            action = float(action) + np.random.normal(0, self.noise_std)

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

    def _render_box(self, box, screen, color_channel):

        if isinstance(color_channel, str):
            color_channel = {"RED": 0, "GREEN": 1, "BLUE": 2}[color_channel]

        y1, x1, y2, x2 = box
        screen_h, screen_w, _ = screen.shape
        y1 = int(screen_h * y1)
        y2 = int(screen_h * y2)
        x1 = int(screen_w * x1)
        x2 = int(screen_w * x2)

        screen[y1:y2, x1:x2, :] = 0  # clear that part of the plot
        screen[y1:y2, x1:x2, color_channel] = 1


    def _render(self, mode="rgb_array", close=False, scale=300):
        screen_h = int(scale * self.height)
        screen_w = int(scale * self.width)

        screen = np.ones(shape=(screen_h, screen_w, 3))  # RGB array
        self._render_box(self.goal_box, screen, "GREEN")

        agent_size = 0.05
        if self.state is not None:
            agent_box = (max(self.state[0] - agent_size / 2, 0),            # y1
                         max(self.state[1] - agent_size / 2, 0),            # x1
                         min(self.state[0] + agent_size / 2, self.height),  # y2
                         min(self.state[1] + agent_size / 2, self.width))   # x2
            self._render_box(agent_box, screen, "BLUE")

        if mode == "human":
            import matplotlib.pyplot as plt
            if self.figure is None:
                plt.ion()
                plt.show()
                self.figure = plt.figure()
                self.ax = self.figure.add_subplot(111)
            self.ax.imshow(screen)
            plt.draw()
            plt.pause(0.1)
        elif mode == "rgb_array":
            return screen


#class PathTrackingTeacher:
#
#    def __init__(self):
