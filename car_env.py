"""CarRacing environment definitions.

Deliberately free of torch imports. On Windows (and macOS) AsyncVectorEnv starts
its workers with `spawn`, so every worker imports the modules that define the
env factories. Keeping torch out of this module keeps the workers light.
"""

import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

FRAME_STACK = 4
FRAME_SIZE = 84


def make_env(render_mode=None):
    """CarRacing with grayscale, resize and frame-stacking wrappers."""
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = FrameStackObservation(env, FRAME_STACK)
    return env


class NoProgressTruncation(gym.Wrapper):
    """TRAINING ONLY. Truncate an episode once the car has gone `patience`
    steps without collecting a new track tile (positive reward).

    The underlying CarRacing configuration is not modified; this only ends
    hopeless episodes early (car stuck on grass, spinning, driving
    backwards) so that more of the step budget is spent on useful driving.
    The cut is a truncation, so the TD target bootstraps from it and the
    learned values are not biased.
    """

    def __init__(self, env, patience=100, grace=50):
        super().__init__(env)
        self.patience = patience
        self.grace = grace  # ignore the zoom-in animation at episode start
        self._t = 0
        self._since_progress = 0

    def reset(self, **kwargs):
        self._t = 0
        self._since_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if reward > 0:
            self._since_progress = 0
        elif self._t > self.grace:
            self._since_progress += 1
        if self._since_progress >= self.patience and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


def make_train_env(patience=100):
    env = make_env()
    if patience:
        env = NoProgressTruncation(env, patience=patience)
    return env
