"""Synthetic 7-DoF dual-camera environment.

This environment is intentionally simple and deterministic enough for smoke-testing
policy rollout and online adaptation pipelines.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


class SyntheticDualCam7DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, max_episode_steps: int = 120, image_height: int = 240, image_width: int = 320):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.image_height = image_height
        self.image_width = image_width

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32),
                "pixels": gym.spaces.Dict(
                    {
                        "fixed": gym.spaces.Box(
                            low=0, high=255, shape=(image_height, image_width, 3), dtype=np.uint8
                        ),
                        "handeye": gym.spaces.Box(
                            low=0, high=255, shape=(image_height, image_width, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )

        self._rng = np.random.default_rng(0)
        self._step = 0
        self._state = np.zeros(7, dtype=np.float32)
        self._target = np.zeros(7, dtype=np.float32)

    def _render_camera(self, camera: str) -> np.ndarray:
        y = np.linspace(0.0, 1.0, self.image_height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, self.image_width, dtype=np.float32)[None, :]

        base = (x + y) * 0.5
        phase = float(self._step * 0.05)
        cam_shift = 0.15 if camera == "handeye" else 0.0
        state_factor = float(np.tanh(self._state.mean()))

        r = np.clip((base + phase + cam_shift + state_factor) * 255.0, 0.0, 255.0)
        g = np.clip((1.0 - base + 0.5 * state_factor) * 255.0, 0.0, 255.0)
        b = np.clip(
            (0.5 + 0.5 * np.sin((x * 8.0) + (y * 6.0) + phase + cam_shift)) * 255.0, 0.0, 255.0
        )
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def _observation(self) -> dict:
        return {
            "agent_pos": self._state.astype(np.float32),
            "pixels": {
                "fixed": self._render_camera("fixed"),
                "handeye": self._render_camera("handeye"),
            },
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._state = self._rng.uniform(-0.5, 0.5, size=(7,)).astype(np.float32)
        self._target = self._rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        self._step += 1
        action = np.asarray(action, dtype=np.float32).reshape(7)
        self._state = np.clip(self._state + 0.06 * action, -1.5, 1.5)

        dist = float(np.linalg.norm(self._state - self._target))
        reward = float(np.exp(-dist))
        terminated = dist < 0.2
        truncated = self._step >= self.max_episode_steps

        return self._observation(), reward, terminated, truncated, {"distance_to_target": dist}

    def render(self):
        return self._render_camera("fixed")

    def close(self):
        return None


register(
    id="gym_synthetic_bimanual/SyntheticDualCam7D-v0",
    entry_point="gym_synthetic_bimanual.env:SyntheticDualCam7DEnv",
)
