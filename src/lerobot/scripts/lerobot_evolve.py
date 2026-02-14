#!/usr/bin/env python

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
"""EVOLVE-style test-time adaptation pipeline.

This script implements the main mechanisms from EVOLVE-VLA:
1. Progress estimation from environment feedback.
2. Accumulative progress reward with milestone/check intervals.
3. Progressive horizon extension curriculum.
4. Online policy updates during deployment.

Usage example:
```
lerobot-evolve \
    --policy.path=outputs/train/your_model/checkpoints/050000/pretrained_model \
    --env.type=pusht \
    --policy.device=cuda \
    --num_stages=3 \
    --iterations_per_stage=10 \
    --rollouts_per_iteration=4
```
"""

import inspect
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.evolve import EvolvePipelineConfig
from lerobot.configs.types import FeatureType
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class StepRecord:
    observation: dict[str, torch.Tensor]
    action: torch.Tensor
    env_reward: float
    progress_reward: float


@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    env_return: float
    progress_return: float
    final_progress: float
    terminated: bool
    truncated: bool
    progress_terminated: bool


class AccumulativeProgressTracker:
    def __init__(self, cfg: EvolvePipelineConfig, image_key: str | None):
        self.cfg = cfg
        self.image_key = image_key
        self.current_progress = 0.0
        self.milestones: list[torch.Tensor] = []

    def reset(self, initial_image: torch.Tensor | None) -> None:
        self.current_progress = 0.0
        self.milestones.clear()
        if initial_image is not None:
            self.milestones.append(initial_image)

    def _map_env_reward(self, env_reward: float) -> float:
        p_cfg = self.cfg.progress
        normalized = (env_reward - p_cfg.env_reward_min) / (p_cfg.env_reward_max - p_cfg.env_reward_min)
        return float(np.clip(normalized, 0.0, 1.0) * 100.0)

    @staticmethod
    def _pixel_cosine_score(reference_image: torch.Tensor, current_image: torch.Tensor) -> float:
        reference_vector = reference_image.float().reshape(-1)
        current_vector = current_image.float().reshape(-1)
        score = torch.nn.functional.cosine_similarity(reference_vector, current_vector, dim=0).item()
        return float(np.clip((score + 1.0) * 50.0, 0.0, 100.0))

    def update(
        self,
        step_index: int,
        current_image: torch.Tensor | None,
        env_reward: float,
    ) -> tuple[float, float, bool]:
        p_cfg = self.cfg.progress
        if current_image is not None and (step_index % p_cfg.milestone_interval == 0):
            self.milestones.append(current_image)

        if step_index % p_cfg.check_interval != 0:
            return 0.0, self.current_progress, False

        if p_cfg.mode == "pixel_cosine" and current_image is not None and len(self.milestones) > 0:
            incremental_score = self._pixel_cosine_score(self.milestones[-1], current_image)
        elif p_cfg.mode == "pixel_cosine" and not p_cfg.use_env_reward_fallback:
            raise RuntimeError(
                "Progress mode is pixel_cosine but no image is available for scoring "
                "(and progress.use_env_reward_fallback is false)."
            )
        else:
            incremental_score = self._map_env_reward(env_reward)

        previous = self.current_progress
        self.current_progress = previous + (100.0 - previous) * incremental_score / 100.0
        progress_delta = (self.current_progress - previous) / 100.0
        should_stop = (self.current_progress / 100.0) >= p_cfg.terminate_threshold
        return progress_delta, self.current_progress, should_stop


def _flatten_envs(envs: dict[str, dict[int, gym.vector.VectorEnv]]) -> list[tuple[str, int, gym.vector.VectorEnv]]:
    tasks: list[tuple[str, int, gym.vector.VectorEnv]] = []
    for task_group in sorted(envs):
        for task_id in sorted(envs[task_group]):
            tasks.append((task_group, task_id, envs[task_group][task_id]))
    return tasks


def _preprocess_for_policy(
    env: gym.vector.VectorEnv,
    raw_observation: dict[str, np.ndarray],
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> dict[str, torch.Tensor]:
    processed = preprocess_observation(raw_observation)
    processed = add_envs_task(env, processed)
    processed = env_preprocessor(processed)
    processed = preprocessor(processed)
    return processed


def _snapshot_policy_observation(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        if not isinstance(value, torch.Tensor):
            continue
        item = value.detach()
        if item.ndim > 0 and item.shape[0] == 1:
            item = item[0]
        snapshot[key] = item.cpu().clone()
    return snapshot


def _resolve_progress_image_key(
    cfg: EvolvePipelineConfig,
    policy: PreTrainedPolicy,
    observation: dict[str, torch.Tensor],
) -> str | None:
    if cfg.progress.image_key is not None:
        if cfg.progress.image_key not in observation:
            raise KeyError(
                f"progress.image_key='{cfg.progress.image_key}' not found in observation keys: "
                f"{sorted(observation.keys())}"
            )
        return cfg.progress.image_key

    for key in policy.config.image_features:
        if key in observation:
            return key

    for key in observation:
        if key.startswith(f"{OBS_IMAGES}."):
            return key
    if OBS_IMAGE in observation:
        return OBS_IMAGE

    return None


def _extract_progress_image(observation: dict[str, torch.Tensor], image_key: str | None) -> torch.Tensor | None:
    if image_key is None:
        return None
    if image_key not in observation:
        return None

    image = observation[image_key].detach()
    if image.ndim > 0 and image.shape[0] == 1:
        image = image[0]
    return image.cpu().clone()


def _infer_action_chunk_len(cfg: EvolvePipelineConfig) -> int:
    if cfg.action_chunk_len is not None:
        return cfg.action_chunk_len

    for field_name in ("chunk_size", "horizon", "n_action_steps"):
        value = getattr(cfg.policy, field_name, None)
        if isinstance(value, int) and value > 0:
            return value

    return 1


def _infer_obs_history_len(cfg: EvolvePipelineConfig) -> int:
    if cfg.obs_history_len is not None:
        return cfg.obs_history_len
    n_obs_steps = getattr(cfg.policy, "n_obs_steps", 1)
    return n_obs_steps if isinstance(n_obs_steps, int) and n_obs_steps > 0 else 1


def _infer_history_keys(cfg: EvolvePipelineConfig) -> set[str]:
    history_keys: set[str] = set()
    if cfg.policy.input_features:
        for key, feature in cfg.policy.input_features.items():
            if feature.type in {FeatureType.STATE, FeatureType.VISUAL, FeatureType.ENV}:
                history_keys.add(key)

    if not history_keys:
        history_keys.update({OBS_STATE, OBS_ENV_STATE})
        history_keys.update(cfg.policy.image_features.keys())

    return history_keys


def _normalize_advantages(rollout_scores: list[float]) -> np.ndarray:
    values = np.asarray(rollout_scores, dtype=np.float32)
    if values.size == 0:
        return values
    std = values.std()
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / (std + 1e-6)


def _build_observation_with_history(
    steps: list[StepRecord],
    step_index: int,
    obs_history_len: int,
    history_keys: set[str],
) -> dict[str, torch.Tensor]:
    current_observation = steps[step_index].observation
    if obs_history_len == 1:
        return {key: value.clone() for key, value in current_observation.items()}

    indexed_steps = []
    for i in range(step_index - obs_history_len + 1, step_index + 1):
        clamped = min(max(i, 0), step_index)
        indexed_steps.append(steps[clamped])

    observation: dict[str, torch.Tensor] = {}
    for key, current_value in current_observation.items():
        if key not in history_keys:
            observation[key] = current_value.clone()
            continue

        stacked_values = [hist_step.observation[key] for hist_step in indexed_steps if key in hist_step.observation]
        if len(stacked_values) != len(indexed_steps):
            observation[key] = current_value.clone()
            continue
        observation[key] = torch.stack(stacked_values, dim=0)

    return observation


def _build_action_chunk(steps: list[StepRecord], step_index: int, chunk_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    action_chunk: list[torch.Tensor] = []
    is_pad: list[bool] = []
    last_action = steps[-1].action
    for offset in range(chunk_len):
        idx = step_index + offset
        if idx < len(steps):
            action_chunk.append(steps[idx].action)
            is_pad.append(False)
        else:
            action_chunk.append(last_action)
            is_pad.append(True)

    return torch.stack(action_chunk, dim=0), torch.tensor(is_pad, dtype=torch.bool)


def _build_training_samples(
    episodes: list[EpisodeRecord],
    advantages: np.ndarray,
    action_chunk_len: int,
    obs_history_len: int,
    history_keys: set[str],
) -> list[dict[str, torch.Tensor]]:
    samples: list[dict[str, torch.Tensor]] = []

    for episode_index, episode in enumerate(episodes):
        if len(episode.steps) == 0:
            continue
        advantage_value = float(advantages[episode_index])
        for step_index in range(len(episode.steps)):
            observation = _build_observation_with_history(
                episode.steps,
                step_index=step_index,
                obs_history_len=obs_history_len,
                history_keys=history_keys,
            )
            action_chunk, action_is_pad = _build_action_chunk(
                episode.steps,
                step_index=step_index,
                chunk_len=action_chunk_len,
            )
            sample = {
                **observation,
                ACTION: action_chunk,
                "action_is_pad": action_is_pad,
                "actions_id_pad": action_is_pad,
                "__advantage__": torch.tensor(advantage_value, dtype=torch.float32),
            }
            samples.append(sample)

    return samples


def _collate_batch(samples: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        batch[key] = torch.stack(values, dim=0).to(device)
    return batch


def _run_policy_updates(
    policy: PreTrainedPolicy,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, torch.Tensor]],
    cfg: EvolvePipelineConfig,
) -> dict[str, float]:
    if len(samples) == 0:
        return {
            "updated": 0.0,
            "loss": 0.0,
            "grad_norm": 0.0,
            "mean_weight": 0.0,
            "supports_per_sample_loss": 0.0,
        }

    supports_per_sample_loss = "reduction" in inspect.signature(policy.forward).parameters
    if not supports_per_sample_loss:
        logging.warning(
            "Policy.forward does not expose a `reduction` argument. "
            "Falling back to unweighted scalar updates."
        )

    policy.train()
    device = torch.device(policy.config.device)
    opt_cfg = cfg.optimization

    loss_values: list[float] = []
    grad_norm_values: list[float] = []
    mean_weight_values: list[float] = []

    for _ in range(opt_cfg.updates_per_iteration):
        sample_indices = np.random.randint(0, len(samples), size=opt_cfg.batch_size)
        minibatch_samples = [samples[i] for i in sample_indices]
        batch = _collate_batch(minibatch_samples, device)
        advantages = batch.pop("__advantage__")

        optimizer.zero_grad(set_to_none=True)
        if supports_per_sample_loss:
            per_sample_loss, _ = policy.forward(batch, reduction="none")
            if per_sample_loss.ndim == 0:
                per_sample_loss = per_sample_loss.repeat(advantages.shape[0])
            elif per_sample_loss.ndim > 1:
                per_sample_loss = per_sample_loss.reshape(per_sample_loss.shape[0], -1).mean(dim=1)
        else:
            scalar_loss, _ = policy.forward(batch)
            if scalar_loss.ndim == 0:
                per_sample_loss = scalar_loss.repeat(advantages.shape[0])
            elif scalar_loss.shape[0] == advantages.shape[0]:
                per_sample_loss = scalar_loss
            else:
                per_sample_loss = scalar_loss.reshape(advantages.shape[0], -1).mean(dim=1)

        weights = torch.exp(opt_cfg.adv_temperature * advantages).clamp(max=opt_cfg.max_weight)
        weights = weights / (weights.mean() + 1e-6)
        loss = (per_sample_loss * weights).mean()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), opt_cfg.grad_clip_norm)
        optimizer.step()

        loss_values.append(loss.item())
        grad_norm_values.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))
        mean_weight_values.append(weights.mean().item())

    return {
        "updated": 1.0,
        "loss": float(np.mean(loss_values)),
        "grad_norm": float(np.mean(grad_norm_values)),
        "mean_weight": float(np.mean(mean_weight_values)),
        "supports_per_sample_loss": 1.0 if supports_per_sample_loss else 0.0,
    }


def _run_rollout(
    cfg: EvolvePipelineConfig,
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    max_steps: int,
    seed: int | None,
    image_key: str | None,
) -> EpisodeRecord:
    seed_list = [seed] if seed is not None else None
    policy.reset()

    raw_observation, _ = env.reset(seed=seed_list)
    observation = _preprocess_for_policy(env, raw_observation, env_preprocessor, preprocessor)
    tracker = AccumulativeProgressTracker(cfg=cfg, image_key=image_key)
    tracker.reset(_extract_progress_image(observation, image_key))

    steps: list[StepRecord] = []
    env_return = 0.0
    progress_return = 0.0
    terminated = False
    truncated = False
    progress_terminated = False

    device = torch.device(policy.config.device)

    for step_index in range(max_steps):
        observation_snapshot = _snapshot_policy_observation(observation)

        autocast_context = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        with torch.inference_mode(), autocast_context:
            policy_action = policy.select_action(observation)
        if cfg.exploration_noise_std > 0:
            policy_action = policy_action + torch.randn_like(policy_action) * cfg.exploration_noise_std
        if cfg.exploration_action_clip is not None:
            policy_action = policy_action.clamp(-cfg.exploration_action_clip, cfg.exploration_action_clip)

        action_for_env = postprocessor(policy_action)
        action_transition = env_postprocessor({ACTION: action_for_env})
        action_for_env = action_transition[ACTION]

        next_observation_raw, reward, terminated_arr, truncated_arr, _ = env.step(
            action_for_env.to("cpu").numpy()
        )
        env_reward = float(np.asarray(reward).reshape(-1)[0])

        next_observation = _preprocess_for_policy(env, next_observation_raw, env_preprocessor, preprocessor)
        progress_delta, current_progress, should_stop = tracker.update(
            step_index=step_index + 1,
            current_image=_extract_progress_image(next_observation, image_key),
            env_reward=env_reward,
        )

        steps.append(
            StepRecord(
                observation=observation_snapshot,
                action=(
                    policy_action.detach().cpu()[0].clone()
                    if policy_action.ndim > 1 and policy_action.shape[0] == 1
                    else policy_action.detach().cpu().clone()
                ),
                env_reward=env_reward,
                progress_reward=progress_delta,
            )
        )

        env_return += env_reward
        progress_return += progress_delta
        observation = next_observation

        terminated = bool(np.asarray(terminated_arr).reshape(-1)[0])
        truncated = bool(np.asarray(truncated_arr).reshape(-1)[0])
        progress_terminated = should_stop
        if terminated or truncated or should_stop:
            break

    return EpisodeRecord(
        steps=steps,
        env_return=env_return,
        progress_return=progress_return,
        final_progress=tracker.current_progress,
        terminated=terminated,
        truncated=truncated,
        progress_terminated=progress_terminated,
    )


def _save_policy_bundle(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


@parser.wrap()
def evolve(cfg: EvolvePipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    envs = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )
    tasks = _flatten_envs(envs)
    if len(tasks) == 0:
        raise RuntimeError("No environments were created.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.to(device)

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
    except Exception as exc:
        if cfg.policy.pretrained_path is None:
            raise
        logging.warning(
            "Failed to load processor files from pretrained path (%s). "
            "Falling back to freshly built processors from policy config.",
            cfg.policy.pretrained_path,
        )
        logging.warning("Original processor loading error: %s", exc)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            preprocessor_overrides=preprocessor_overrides,
        )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg.optimization.learning_rate,
        weight_decay=cfg.optimization.weight_decay,
    )

    # Resolve image key once from an initial processed observation.
    _, _, probe_env = tasks[0]
    probe_observation_raw, _ = probe_env.reset(seed=[cfg.seed])
    probe_observation = _preprocess_for_policy(probe_env, probe_observation_raw, env_preprocessor, preprocessor)
    image_key = _resolve_progress_image_key(cfg, policy, probe_observation)
    if cfg.progress.mode == "pixel_cosine" and image_key is None and not cfg.progress.use_env_reward_fallback:
        raise RuntimeError(
            "Unable to resolve an image key for pixel_cosine mode and no env-reward fallback is allowed."
        )

    action_chunk_len = _infer_action_chunk_len(cfg)
    obs_history_len = _infer_obs_history_len(cfg)
    history_keys = _infer_history_keys(cfg)
    logging.info(
        "Online adaptation batch spec: "
        f"action_chunk_len={action_chunk_len}, obs_history_len={obs_history_len}"
    )

    metrics_per_iteration: list[dict[str, Any]] = []
    global_iteration = 0
    rollout_seed = cfg.seed

    try:
        for stage_idx in range(cfg.num_stages):
            max_steps = min(cfg.initial_max_steps + stage_idx * cfg.horizon_increment, cfg.max_steps_cap)
            logging.info(
                colored(
                    f"Stage {stage_idx + 1}/{cfg.num_stages} | max_steps={max_steps}",
                    "cyan",
                    attrs=["bold"],
                )
            )

            for stage_iter in range(cfg.iterations_per_stage):
                episodes: list[EpisodeRecord] = []
                for rollout_idx in range(cfg.rollouts_per_iteration):
                    task_group, task_id, env = tasks[
                        (global_iteration * cfg.rollouts_per_iteration + rollout_idx) % len(tasks)
                    ]
                    episode = _run_rollout(
                        cfg=cfg,
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        max_steps=max_steps,
                        seed=rollout_seed,
                        image_key=image_key,
                    )
                    rollout_seed += 1
                    episodes.append(episode)
                    logging.info(
                        f"Rollout [{task_group}:{task_id}] "
                        f"len={len(episode.steps)} env_return={episode.env_return:.4f} "
                        f"progress_return={episode.progress_return:.4f} "
                        f"final_progress={episode.final_progress:.2f}"
                    )

                rollout_scores = [
                    episode.progress_return if cfg.train_reward_source == "progress" else episode.env_return
                    for episode in episodes
                ]
                advantages = _normalize_advantages(rollout_scores)

                samples = _build_training_samples(
                    episodes=episodes,
                    advantages=advantages,
                    action_chunk_len=action_chunk_len,
                    obs_history_len=obs_history_len,
                    history_keys=history_keys,
                )
                update_metrics = _run_policy_updates(policy, optimizer, samples, cfg)

                mean_env_return = float(np.mean([ep.env_return for ep in episodes])) if episodes else 0.0
                mean_progress_return = (
                    float(np.mean([ep.progress_return for ep in episodes])) if episodes else 0.0
                )
                mean_final_progress = (
                    float(np.mean([ep.final_progress for ep in episodes])) if episodes else 0.0
                )
                success_rate = float(
                    np.mean(
                        [
                            ep.final_progress >= (cfg.progress.terminate_threshold * 100.0)
                            for ep in episodes
                        ]
                    )
                    * 100.0
                )
                mean_episode_len = float(np.mean([len(ep.steps) for ep in episodes])) if episodes else 0.0

                iteration_metrics = {
                    "stage": stage_idx + 1,
                    "stage_iteration": stage_iter + 1,
                    "global_iteration": global_iteration + 1,
                    "max_steps": max_steps,
                    "rollouts": len(episodes),
                    "samples": len(samples),
                    "mean_env_return": mean_env_return,
                    "mean_progress_return": mean_progress_return,
                    "mean_final_progress": mean_final_progress,
                    "success_rate": success_rate,
                    "mean_episode_len": mean_episode_len,
                    "update_loss": update_metrics["loss"],
                    "update_grad_norm": update_metrics["grad_norm"],
                    "update_mean_weight": update_metrics["mean_weight"],
                    "supports_per_sample_loss": bool(update_metrics["supports_per_sample_loss"]),
                }
                metrics_per_iteration.append(iteration_metrics)
                global_iteration += 1

                logging.info(
                    f"Iter {iteration_metrics['global_iteration']}: "
                    f"env_return={mean_env_return:.4f}, "
                    f"progress_return={mean_progress_return:.4f}, "
                    f"success={success_rate:.1f}%, "
                    f"samples={len(samples)}, "
                    f"update_loss={update_metrics['loss']:.6f}"
                )

            stage_dir = output_dir / f"stage_{stage_idx + 1:02d}_policy"
            _save_policy_bundle(policy, preprocessor, postprocessor, stage_dir)

        final_policy_dir = output_dir / "adapted_policy"
        _save_policy_bundle(policy, preprocessor, postprocessor, final_policy_dir)

        metrics_payload = {
            "config": asdict(cfg),
            "iterations": metrics_per_iteration,
            "final_policy_dir": str(final_policy_dir),
            "resolved_progress_image_key": image_key,
            "action_chunk_len": action_chunk_len,
            "obs_history_len": obs_history_len,
        }
        with open(output_dir / "evolve_metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2, default=str)

        logging.info(f"Saved adapted policy to {final_policy_dir}")
        logging.info("End of evolve")
    finally:
        close_envs(envs)


def main():
    register_third_party_plugins()
    init_logging()
    evolve()


if __name__ == "__main__":
    main()
