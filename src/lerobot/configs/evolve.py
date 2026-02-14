# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

logger = getLogger(__name__)


@dataclass
class EvolveProgressConfig:
    # Mode choices:
    # - "pixel_cosine": cosine similarity between milestone and current image
    # - "env_reward": map environment reward to [0, 100]
    mode: str = "pixel_cosine"
    # Optional override for the image key used by pixel_cosine mode.
    image_key: str | None = None
    # Evaluate progress every `check_interval` steps.
    check_interval: int = 8
    # Insert a new milestone every `milestone_interval` steps.
    milestone_interval: int = 64
    # Stop rollout early if accumulated progress >= terminate_threshold.
    terminate_threshold: float = 0.95
    # Reward range used when mode == "env_reward".
    env_reward_min: float = -1.0
    env_reward_max: float = 1.0
    # In pixel mode, if image key cannot be resolved, fallback to env reward mapping.
    use_env_reward_fallback: bool = True

    def validate(self) -> None:
        if self.check_interval < 1:
            raise ValueError("progress.check_interval must be >= 1.")
        if self.milestone_interval < 1:
            raise ValueError("progress.milestone_interval must be >= 1.")
        if self.check_interval > self.milestone_interval:
            raise ValueError(
                "progress.check_interval must be <= progress.milestone_interval so milestones are meaningful."
            )
        if not 0.0 < self.terminate_threshold <= 1.0:
            raise ValueError("progress.terminate_threshold must be in (0, 1].")
        if self.env_reward_max <= self.env_reward_min:
            raise ValueError("progress.env_reward_max must be strictly greater than progress.env_reward_min.")
        if self.mode not in {"pixel_cosine", "env_reward"}:
            raise ValueError("progress.mode must be one of {'pixel_cosine', 'env_reward'}.")


@dataclass
class EvolveOptimizationConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    # Number of gradient updates after each rollout iteration.
    updates_per_iteration: int = 16
    # Minibatch size sampled from the collected trajectories.
    batch_size: int = 16
    grad_clip_norm: float = 1.0
    # Advantage-to-weight transform: weight = exp(adv_temperature * advantage).
    adv_temperature: float = 1.0
    max_weight: float = 20.0

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("optimization.learning_rate must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("optimization.weight_decay must be >= 0.")
        if self.updates_per_iteration < 1:
            raise ValueError("optimization.updates_per_iteration must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("optimization.batch_size must be >= 1.")
        if self.grad_clip_norm <= 0:
            raise ValueError("optimization.grad_clip_norm must be > 0.")
        if self.max_weight <= 0:
            raise ValueError("optimization.max_weight must be > 0.")


@dataclass
class EvolvePipelineConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int = 1000
    # Rename map for observation key adaptation.
    rename_map: dict[str, str] = field(default_factory=dict)
    # Explicit consent to execute remote code from the Hub (required for hub envs).
    trust_remote_code: bool = False

    # Exploration during rollout.
    exploration_noise_std: float = 0.0
    exploration_action_clip: float | None = None

    # Progressive horizon extension.
    num_stages: int = 3
    iterations_per_stage: int = 20
    rollouts_per_iteration: int = 4
    initial_max_steps: int = 64
    horizon_increment: int = 32
    max_steps_cap: int = 256

    # Which reward drives adaptation objective:
    # - "progress": paper-like accumulative progress reward
    # - "env": raw env return
    train_reward_source: str = "progress"
    # Optional overrides for constructing online adaptation batches.
    action_chunk_len: int | None = None
    obs_history_len: int | None = None

    progress: EvolveProgressConfig = field(default_factory=EvolveProgressConfig)
    optimization: EvolveOptimizationConfig = field(default_factory=EvolveOptimizationConfig)

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)
        else:
            logger.warning(
                "No pretrained policy path was provided. The policy will be created from scratch."
            )

        if self.policy is None:
            raise ValueError("Policy is not configured. Please specify a pretrained policy with `--policy.path`.")

        if not self.job_name:
            self.job_name = f"{self.env.type}_{self.policy.type}_evolve"
            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            evolve_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/evolve") / evolve_dir

        self.validate()

    def validate(self) -> None:
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1.")
        if self.iterations_per_stage < 1:
            raise ValueError("iterations_per_stage must be >= 1.")
        if self.rollouts_per_iteration < 1:
            raise ValueError("rollouts_per_iteration must be >= 1.")
        if self.initial_max_steps < 1:
            raise ValueError("initial_max_steps must be >= 1.")
        if self.horizon_increment < 0:
            raise ValueError("horizon_increment must be >= 0.")
        if self.max_steps_cap < self.initial_max_steps:
            raise ValueError("max_steps_cap must be >= initial_max_steps.")
        if self.exploration_noise_std < 0:
            raise ValueError("exploration_noise_std must be >= 0.")
        if self.exploration_action_clip is not None and self.exploration_action_clip <= 0:
            raise ValueError("exploration_action_clip must be > 0 when set.")
        if self.train_reward_source not in {"progress", "env"}:
            raise ValueError("train_reward_source must be one of {'progress', 'env'}.")
        if self.action_chunk_len is not None and self.action_chunk_len < 1:
            raise ValueError("action_chunk_len must be >= 1 when set.")
        if self.obs_history_len is not None and self.obs_history_len < 1:
            raise ValueError("obs_history_len must be >= 1 when set.")

        self.progress.validate()
        self.optimization.validate()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]
