import os
import time
import pickle
import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm
from collections import deque

from log.wandb_logger import WandbLogger
from policy.base import Base

from utils.sampler import OnlineSampler


# model-free policy trainer
class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        timesteps: int = 1e6,
        episode_len:int = 200,
        log_interval: int = 2,
        eval_num: int = 10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.timesteps = timesteps
        self.episode_len = episode_len
        self.nupdates = self.policy.nupdates

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5

        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Train loop
        eval_idx = 0
        with tqdm(
            total=self.timesteps, desc=f"{self.policy.name} Training (Timesteps)"
        ) as pbar:
            while pbar.n < self.timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division

                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )

                loss_dict, ppo_timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(ppo_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / step
                remaining_time = avg_time_per_iter * (self.timesteps - step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, render_imgs = self.evaluate()

                    # Manual logging
                    self.write_log(eval_dict, step=step, eval_log=True)
                    if eval_idx % 25 == 0:
                        self.write_video(
                            render_imgs,
                            step=step,
                            logdir=f"{self.policy.name}",
                            name="rendering",
                        )

                    self.last_reward_mean.append(eval_dict[f"eval/rew_mean"])
                    self.last_reward_std.append(eval_dict[f"eval/rew_std"])

                    self.save_model(step)

            torch.cuda.empty_cache()

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def evaluate(self):
        """
        Given one ref, show tracking performance
        """
        ep_buffer = []
        rendering_imgs = []
        for num_episodes in range(self.eval_num):
            ep_reward, ep_success, ep_control_effort = 0, 0, 0

            # Env initialization
            obs, infos = self.env.reset(seed=self.seed)
            for t in range(self.episode_len):
                with torch.no_grad():
                    a, _ = self.policy(obs, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                next_obs, rew, term, trunc, infos = self.env.step(a)
                done = True if term or trunc or t == (self.episode_len - 1) else False

                obs = next_obs
                ep_reward += rew
                ep_control_effort += np.linalg.norm(a)
                ep_success = np.maximum(ep_success, infos["success"])
                if num_episodes == 0:
                    rendering_imgs.append(self.env.render())

                if done:
                    ep_buffer.append(
                        {
                            "reward": ep_reward,
                            "success": ep_success,
                            "control_effort": ep_control_effort,
                        }
                    )

                    break

        rew_list = [ep_info["reward"] for ep_info in ep_buffer]
        suc_list = [ep_info["success"] for ep_info in ep_buffer]
        ctr_list = [ep_info["control_effort"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(rew_list), np.std(rew_list)
        suc_mean, suc_std = np.mean(suc_list), np.std(suc_list)
        ctr_mean, ctr_std = np.mean(ctr_list), np.std(ctr_list)

        eval_dict = {
            f"eval/rew_mean": rew_mean,
            f"eval/rew_std": rew_std,
            f"eval/suc_mean": suc_mean,
            f"eval/suc_std": suc_std,
            f"eval/ctr_effort_mean": ctr_mean,
            f"eval/ctr_effort_std": ctr_std,
        }

        return eval_dict, rendering_imgs

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        path_image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=path_image_path)

    def write_video(self, rendering_imgs: list, step: int, logdir: str, name: str):
        path_render_path = os.path.join(logdir, name)
        try:
            self.logger.write_videos(
                step=step, images=rendering_imgs, logdir=path_render_path
            )
        except:
            print("Video logging error. Likely a system problem.")

    def save_model(self, e):
        # save checkpoint
        name = f"model_{e}.p"
        path = os.path.join(self.logger.checkpoint_dir, name)
        pickle.dump(
            (self.policy),
            open(path, "wb"),
        )

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            name = f"best_model_{e}.p"
            path = os.path.join(self.logger.log_dir, name)
            pickle.dump(
                (self.policy),
                open(path, "wb"),
            )

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict
