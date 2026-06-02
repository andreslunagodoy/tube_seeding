import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ThreePositionEnv(gym.Env):
    """
    A minimal 1D environment with three positions at coordinates -1, 0, and 1.

    - Observation: the current coordinate as a 1D array with shape (1,), values in {-1, 0, 1}
    - Actions: Discrete(2) -> 0: move left, 1: move right
    - Reward: +1 if agent is at coordinate 1 after the action; 0 otherwise
    - Invalid actions at boundaries are masked by get_action_mask() but still allowed;
      they result in staying at the boundary.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, max_episode_steps: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps  # if None, episodes don't auto-truncate

        # Two actions: 0 -> left, 1 -> right
        self.action_space = spaces.Discrete(2)

        # Observation is the coordinate as a single integer in [-1, 1]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)

        self.position: int = -1
        self._elapsed_steps: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.position = -1
        self._elapsed_steps = 0
        obs = np.array([self.position], dtype=np.int8)
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        # Apply action with boundary conditions
        if action == 0:  # move left
            self.position = max(-1, self.position - 1)
        elif action == 1:  # move right
            self.position = min(1, self.position + 1)

        reward = 1.0 if self.position == 1 else 0.0

        self._elapsed_steps += 1
        terminated = False  # This task doesn't have a terminal state by default
        truncated = False
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        obs = np.array([self.position], dtype=np.int8)
        info = {"action_mask": self.get_action_mask()}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a mask over actions indicating which are valid at the current state.

        - mask[0] (left) is 0 when position == -1, else 1
        - mask[1] (right) is 0 when position == 1, else 1
        """
        left_valid = True if self.position > -1 else False
        right_valid = True if self.position < 1 else False
        return np.array([left_valid, right_valid], dtype=np.bool)

    def render(self):
        track = " ".join([f"[{'*' if p == self.position else ' '}{p}{'*' if p == self.position else ' '}]" for p in (-1, 0, 1)])
        print(f"Position: {self.position}  {track}")

    def close(self):
        pass

# Register the environment
gym.register(
    id='ThreePositionEnv-v0',
    entry_point=__name__ + ':ThreePositionEnv'
)

def test_env():
    # Quick manual test
    # env = ThreePositionEnv(max_episode_steps=10, render_mode="human")
    env = gym.make('ThreePositionEnv-v0', max_episode_steps=10, render_mode="human")
    obs, info = env.reset()
    print("Initial obs:", obs, "mask:", info["action_mask"])
    done = False
    truncated = False
    while not (done or truncated):
        # Choose a valid action using the mask (prefer moving right)
        mask = info["action_mask"]
        action = 1 if mask[1] == 1 else 0
        obs, reward, done, truncated, info = env.step(action)
        print("Obs:", obs, "Reward:", reward, "Mask:", info["action_mask"])
    env.close()
