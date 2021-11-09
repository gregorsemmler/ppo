import argparse
import logging
import math
from collections import deque
from datetime import datetime
from os import makedirs
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from data import EnvironmentsDataset, Policy
from model import ActorCriticModel
from play import play_environment
from common import save_checkpoint, load_checkpoint, GracefulExit, get_action_space_details, get_model, \
    get_preprocessor, get_environment

logger = logging.getLogger(__name__)


class DummySummaryWriter(object):

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass


class ActorCriticReturnScheduler(object):

    def __init__(self, optimizer, milestones, factor=0.1, critic_optimizer=None):
        self.optimizer = optimizer
        self.critic_optimizer = critic_optimizer
        self.milestones = np.sort(milestones)
        self.begin_lr = self.get_lr()
        self.begin_critic_lr = self.get_critic_lr()
        self.factor = factor

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def get_critic_lr(self):
        if self.critic_optimizer is None:
            return None
        for param_group in self.critic_optimizer.param_groups:
            return param_group["lr"]

    def set_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"] = new_lr

    def set_critic_lr(self, new_lr):
        if self.critic_optimizer is not None:
            for g in self.critic_optimizer.param_groups:
                g["lr"] = new_lr

    def step(self, returns):
        idx = 0
        for idx in range(len(self.milestones)):
            if returns <= self.milestones[idx]:
                break

        if idx > 0:
            new_lr = self.begin_lr * self.factor ** idx
            self.set_lr(new_lr)
            if self.critic_optimizer is not None:
                new_critic_lr = self.begin_critic_lr * self.factor ** idx
                self.set_critic_lr(new_critic_lr)
        else:
            self.set_lr(self.begin_lr)
            self.set_critic_lr(self.begin_critic_lr)


class ActorCriticTrainer(object):

    def __init__(self, config, model: ActorCriticModel, model_id, trainer_id=None, optimizer=None,
                 critic_optimizer=None, scheduler=None, checkpoint_path=None, save_optimizer=False, writer=None,
                 batch_wise_scheduler=True, num_mean_results=100, target_mean_returns=None,
                 graceful_exiter: GracefulExit = None, action_limits=None):
        self.value_factor = config.value_factor
        self.policy_factor = config.policy_factor
        self.entropy_factor = config.entropy_factor
        self.max_norm = config.max_norm
        self.lr = config.lr
        self.gamma = config.gamma
        self.undiscounted_log = config.undiscounted_log
        self.log_frequency = config.log_frequency
        self.num_eval_episodes = config.n_eval_episodes
        self.num_mean_results = num_mean_results
        self.target_mean_returns = target_mean_returns

        if config.device_token is None:
            device_token = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_token = config.device_token
        self.device = torch.device(device_token)

        self.trainer_id = "" if trainer_id is None else str(trainer_id)
        self.model = model
        self.shared = model.is_shared
        self.model_id = model_id

        if self.shared:
            self.optimizer = optimizer if optimizer is not None else Adam(model.parameters(), lr=self.lr,
                                                                          weight_decay=config.l2_regularization,
                                                                          eps=config.eps)
        else:
            self.optimizer = optimizer if optimizer is not None else Adam(model.actor_parameters(), lr=self.lr,
                                                                          weight_decay=config.l2_regularization,
                                                                          eps=config.eps)
            if critic_optimizer is not None:
                self.critic_optimizer = critic_optimizer
            else:
                self.critic_optimizer = Adam(model.critic_parameters(), lr=config.critic_lr,
                                             weight_decay=config.critic_l2_regularization, eps=config.critic_eps)

        self.writer = writer if writer is not None else DummySummaryWriter()

        self.discrete = self.model.is_discrete
        self.action_limits = action_limits

        if scheduler is not None:
            self.scheduler = scheduler
        elif config.scheduler_returns is not None:
            self.scheduler = ActorCriticReturnScheduler(self.optimizer, config.scheduler_returns,
                                                        config.scheduler_factor, critic_optimizer=self.critic_optimizer)
        else:
            self.scheduler = None

        self.checkpoint_path = checkpoint_path
        self.save_optimizer = save_optimizer
        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.curr_train_episode_idx = 0
        self.count_episodes = 0
        self.batch_wise_scheduler = batch_wise_scheduler
        self.target_reached = False
        self.last_returns = deque(maxlen=self.num_mean_results)
        self.graceful_exiter = graceful_exiter

    def scheduler_step(self, metrics=None):
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr()
            current_critic_lr = self.scheduler.get_critic_lr()

            self.scheduler.step(metrics)

            log_prefix = "batch" if self.batch_wise_scheduler else "epoch"
            log_idx = self.curr_train_batch_idx if self.batch_wise_scheduler else self.curr_epoch_idx
            self.writer.add_scalar(f"{log_prefix}/lr", current_lr, log_idx)
            if current_critic_lr is not None:
                self.writer.add_scalar(f"{log_prefix}/critic_lr", current_critic_lr, log_idx)

    def save_checkpoint(self, filename=None, best=False):
        if self.checkpoint_path is None:
            return None

        if filename is None:
            filename = f"{self.model_id}_{self.curr_epoch_idx:03d}.tar"

        if best:
            path = join(self.checkpoint_path, "best", filename)
        else:
            path = join(self.checkpoint_path, filename)

        if self.save_optimizer:
            save_checkpoint(path, self.model, optimizer=self.optimizer, critic_optimizer=self.critic_optimizer)
        else:
            save_checkpoint(path, self.model)
        return path

    def fit(self, dataset_train, eval_env, eval_policy, num_epochs=None, training_seed=None):
        if training_seed is not None:
            np.random.seed(training_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(training_seed)

        self.curr_epoch_idx = 0
        self.curr_train_batch_idx = 0
        self.curr_val_batch_idx = 0
        self.curr_train_episode_idx = 0
        self.count_episodes = 0
        self.target_reached = False
        self.last_returns.clear()

        if num_epochs is None:
            logger.info(f"{self.trainer_id}# Starting training.")
        else:
            logger.info(f"{self.trainer_id}# Starting training for {num_epochs} epochs.")

        while num_epochs is None or self.curr_epoch_idx < num_epochs:
            logger.info(f"{self.trainer_id}# Epoch {self.curr_epoch_idx}")
            logger.info(f"{self.trainer_id}# Training")
            self.train(dataset_train)

            if self.target_reached:
                logger.info(f"Reached target mean returns. Ending training.")
                save_path = self.save_checkpoint(best=True)
                if save_path is not None:
                    logger.info(f"Saved model to '{save_path}'")
                break

            if self.graceful_exiter is not None and not self.graceful_exiter.run:
                filename = f"{self.model_id}_{datetime.now():%d%m%Y_%H%M%S}.tar"

                save_path = self.save_checkpoint(filename)
                if save_path is not None:
                    logger.info(f"Saved model to '{save_path}'")
                break

            if self.num_eval_episodes > 0:
                logger.info(f"{self.trainer_id}# Validation")
                play_environment(eval_env, eval_policy, num_episodes=self.num_eval_episodes, gamma=self.gamma)

            if not self.batch_wise_scheduler:
                self.scheduler_step(self.get_mean_returns())
            self.curr_epoch_idx += 1
            self.save_checkpoint()

    def get_mean_returns(self):
        return 0.0 if len(self.last_returns) == 0 else sum(self.last_returns) / len(self.last_returns)

    def train(self, dataset):
        self.model.train()

        ep_l = 0.0
        ep_p_l = 0.0
        ep_v_l = 0.0
        ep_e_l = 0.0
        ep_episode_length = 0.0
        ep_episode_returns = 0.0
        count_batches = 0
        count_epoch_episodes = 0

        for er_returns, batch in dataset.data():
            b_l, p_l, v_l, e_l = self.training_step(batch, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/loss", b_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/policy_loss", p_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/value_loss", v_l, self.curr_train_batch_idx)
            self.writer.add_scalar(f"train_batch/entropy_loss", e_l, self.curr_train_batch_idx)

            batch_ep_len = 0.0
            batch_ep_ret = 0.0

            for length, ret, ret_u in er_returns:
                ep_ret = ret_u if self.undiscounted_log else ret
                self.writer.add_scalar(f"train_batch/episode_length", length,
                                       self.curr_train_episode_idx)
                self.writer.add_scalar(f"train_batch/episode_return", ep_ret,
                                       self.curr_train_episode_idx)
                ep_episode_length += length
                ep_episode_returns += ep_ret
                self.curr_train_episode_idx += 1
                self.count_episodes += 1
                count_epoch_episodes += 1
                batch_ep_len += length
                batch_ep_ret += ep_ret
                self.last_returns.append(ep_ret)

            mean_returns = self.get_mean_returns()
            batch_ep_len = None if len(er_returns) == 0 else batch_ep_len / len(er_returns)
            batch_ep_ret = None if len(er_returns) == 0 else batch_ep_ret / len(er_returns)

            if self.curr_train_batch_idx % self.log_frequency == 0:
                log_msg = f"{self.trainer_id}# Epoch: {self.curr_epoch_idx} Batch: {self.curr_train_batch_idx}: " \
                          f"{self.count_episodes} Episodes, Mean{self.num_mean_results} Returns: {mean_returns:.6g}, " \
                          f"Loss: {b_l:.5g} Policy Loss: {p_l:.5g} Value Loss: {v_l:.5g} Entropy Loss: {e_l:.3g} "

                if batch_ep_len is not None:
                    log_msg += f"Ep Length: {batch_ep_len:.5g} Ep Return: {batch_ep_ret:.5g}"

                logger.info(log_msg)

            self.curr_train_batch_idx += 1
            count_batches += 1
            ep_l += b_l
            ep_p_l += p_l
            ep_v_l += v_l
            ep_e_l += e_l

            if self.target_mean_returns is not None and mean_returns >= self.target_mean_returns \
                    and len(self.last_returns) >= self.num_mean_results:
                self.target_reached = True
                break

            if self.graceful_exiter is not None and not self.graceful_exiter.run:
                break

        ep_l /= max(1.0, count_batches)
        ep_p_l /= max(1.0, count_batches)
        ep_v_l /= max(1.0, count_batches)
        ep_e_l /= max(1.0, count_batches)
        ep_episode_length /= max(1.0, count_epoch_episodes)
        ep_episode_returns /= max(1.0, count_epoch_episodes)

        self.writer.add_scalar(f"train_epoch/loss", ep_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/policy_loss", ep_p_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/value_loss", ep_v_l, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/episode_length", ep_episode_length, self.curr_epoch_idx)
        self.writer.add_scalar(f"train_epoch/episode_return", ep_episode_returns, self.curr_epoch_idx)
        logger.info(f"{self.trainer_id}# Epoch {self.curr_epoch_idx}: Loss: {ep_l:.6g} "
                    f"Policy Loss: {ep_p_l:.6g} Value Loss: {ep_v_l:.6g} Entropy Loss: {ep_e_l:.6g} "
                    f"Episode Length: {ep_episode_length:.6g} Episode Return: {ep_episode_returns:.6g}")

    def calculate_policy_and_entropy_loss(self, actions, advantages, policy_out):
        if self.discrete:
            log_probs_out = F.log_softmax(policy_out, dim=1)
            probs_out = F.softmax(policy_out, dim=1)

            policy_loss = advantages * log_probs_out[range(len(probs_out)), actions]
            policy_loss = self.policy_factor * -policy_loss.mean()
            entropy_loss = self.entropy_factor * (probs_out * log_probs_out).sum(dim=1).mean()
            return policy_loss, entropy_loss

        mean, log_std = policy_out

        if self.action_limits:
            low, high = self.action_limits
            mean = torch.clamp(mean, low, high)
            log_std = torch.clamp(log_std, math.log(1e-5), 2 * math.log(high - low))

        variance = torch.exp(2 * log_std)

        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        # Log of normal distribution:
        log_probs = -((actions - mean) ** 2) / (2 * variance) - log_std - math.log(math.sqrt(2 * math.pi))
        policy_loss = self.policy_factor * -(advantages * log_probs).mean()
        # Entropy of normal distribution:
        entropy_loss = self.entropy_factor * (0.5 + 0.5 * math.log(2 * math.pi) + log_std).mean()
        return policy_loss, entropy_loss

    def training_step(self, batch, batch_idx):
        states_t = torch.cat(batch.states).to(self.device)
        actions = batch.actions
        values_t = torch.FloatTensor(np.array(batch.values)).to(self.device)
        advantages_t = torch.FloatTensor(np.array(batch.advantages)).to(self.device)

        policy_out, value_out = self.model(states_t)

        value_loss = self.value_factor * F.mse_loss(value_out.squeeze(-1), values_t)

        policy_loss, entropy_loss = self.calculate_policy_and_entropy_loss(actions, advantages_t, policy_out)

        if self.shared:
            loss = entropy_loss + value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()

            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()
            if self.batch_wise_scheduler:
                self.scheduler_step(self.get_mean_returns())
        else:
            # Actor
            actor_loss = policy_loss + entropy_loss

            self.optimizer.zero_grad()
            actor_loss.backward()

            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()

            # Critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()

            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.critic_optimizer.step()
            if self.batch_wise_scheduler:
                self.scheduler_step(self.get_mean_returns())

        p_loss, v_loss, e_loss = policy_loss.item(), value_loss.item(), entropy_loss.item()
        total_loss = p_loss + v_loss + e_loss

        return total_loss, p_loss, v_loss, e_loss


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_processes", type=int, default=1)
    parser.add_argument("--n_envs", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--scheduler_returns", type=lambda s: [int(e) for e in s.split(",")])
    parser.add_argument("--scheduler_factor", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--l2_regularization", type=float, default=0)
    parser.add_argument("--critic_eps", type=float, default=1e-3)
    parser.add_argument("--critic_l2_regularization", type=float, default=0)
    parser.add_argument("--epoch_length", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=-1)
    parser.add_argument("--n_mean_results", type=int, default=100)
    parser.add_argument("--target_mean_returns", type=float)
    parser.add_argument("--value_factor", type=float, default=1.0)
    parser.add_argument("--policy_factor", type=float, default=1.0)
    parser.add_argument("--entropy_factor", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=0.1)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--save_optimizer", type=bool, default=False)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--env_name", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--log_frequency", type=int, default=1)
    parser.add_argument("--partial_unroll", dest="partial_unroll", action="store_true")
    parser.add_argument("--no_partial_unroll", dest="partial_unroll", action="store_false")
    parser.add_argument("--undiscounted_log", dest="undiscounted_log", action="store_true")
    parser.add_argument("--no_undiscounted_log", dest="undiscounted_log", action="store_false")
    parser.add_argument("--atari", dest="atari", action="store_true")
    parser.add_argument("--no_atari", dest="atari", action="store_false")
    parser.add_argument("--shared_model", dest="shared_model", action="store_true")
    parser.add_argument("--no_shared_model", dest="shared_model", action="store_false")
    parser.add_argument("--fixed_std", dest="fixed_std", action="store_true")
    parser.add_argument("--no_fixed_std", dest="fixed_std", action="store_false")
    parser.add_argument("--tensorboardlog", dest="tensorboardlog", action="store_true")
    parser.add_argument("--no_tensorboardlog", dest="tensorboardlog", action="store_false")
    parser.add_argument("--graceful_exit", dest="graceful_exit", action="store_true")
    parser.add_argument("--no_graceful_exit", dest="graceful_exit", action="store_false")
    parser.set_defaults(atari=True, partial_unroll=True, graceful_exit=True, undiscounted_log=True, shared_model=False,
                        tensorboardlog=False, fixed_std=True)

    args = parser.parse_args()

    env_name = args.env_name
    atari = args.atari
    checkpoint_path = args.checkpoint_path
    shared_model = args.shared_model
    pretrained_path = args.pretrained_path

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)

    if checkpoint_path is not None:
        best_models_path = join(checkpoint_path, "best")
        makedirs(checkpoint_path, exist_ok=True)
        makedirs(best_models_path, exist_ok=True)

    model = get_model(env_name, shared_model, atari, device, fixed_std=args.fixed_std)

    if pretrained_path is not None:
        load_checkpoint(pretrained_path, model, device=device)
        logger.info(f"Loaded model from '{pretrained_path}'")

    mp.set_start_method("spawn")

    processes = []
    for trainer_id in range(args.n_processes):
        p = mp.Process(target=training, args=(args, model, trainer_id, device))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def training(args, model, trainer_id, device):
    logging.basicConfig(level=logging.INFO)
    env_name = args.env_name
    env_count = args.n_envs
    n_steps = args.n_steps
    gamma = args.gamma
    batch_size = args.batch_size
    atari = args.atari
    epoch_length = args.epoch_length
    target_mean_returns = args.target_mean_returns
    partial_unroll = args.partial_unroll
    checkpoint_path = args.checkpoint_path
    num_epochs = args.n_epochs if args.n_epochs > 0 else None
    num_mean_results = args.n_mean_results
    run_id = args.run_id if args.run_id is not None else f"run_{datetime.now():%d%m%Y_%H%M%S}"
    run_id = f"{run_id}_{trainer_id}"
    model_id = f"{run_id}" if args.model_id is None else args.model_id

    preprocessor = get_preprocessor(env_name, atari)
    environments = [get_environment(env_name, atari) for _ in range(env_count)]

    eval_env = get_environment(env_name, atari)
    _, _, limits = get_action_space_details(eval_env.action_space)
    writer = SummaryWriter(comment=f"-{run_id}") if args.tensorboardlog else DummySummaryWriter()

    dataset = EnvironmentsDataset(environments, model, n_steps, gamma, batch_size, preprocessor, device,
                                  epoch_length=epoch_length, partial_unroll=partial_unroll, action_limits=limits)

    graceful_exiter = GracefulExit() if args.graceful_exit else None
    trainer = ActorCriticTrainer(args, model, model_id, trainer_id=trainer_id, writer=writer,
                                 num_mean_results=num_mean_results, target_mean_returns=target_mean_returns,
                                 checkpoint_path=checkpoint_path, graceful_exiter=graceful_exiter, action_limits=limits)
    eval_policy = Policy(model, preprocessor, device, action_limits=limits)
    trainer.fit(dataset, eval_env, eval_policy, num_epochs=num_epochs)


if __name__ == "__main__":
    main()
