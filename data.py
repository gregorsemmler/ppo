import logging
import math
import uuid
from typing import Sequence

import torch
import torch.nn.functional as F
import numpy as np
from gym import Env
from torch.distributions import Categorical, Normal

from model import ActorCriticModel

logger = logging.getLogger(__name__)


def categorical_action_selector(policy_out, action_limits=None):
    probs = F.softmax(policy_out, dim=1)
    actions = Categorical(probs.detach().cpu()).sample().detach().cpu().numpy()
    return actions if action_limits is None else np.clip(actions, action_limits[0], action_limits[1])


def normal_action_selector(policy_out, action_limits=None):
    mean, log_std = policy_out

    if action_limits is None:
        low, high = action_limits
        mean = torch.clamp(mean, low, high)
        log_std = torch.clamp(log_std, math.log(1e-5), 2 * math.log(high - low))

    std_dev = torch.exp(log_std)
    actions = Normal(mean, std_dev).sample().detach().cpu().numpy()
    return actions if action_limits is None else np.clip(actions, action_limits[0], action_limits[1])


class Policy(object):

    def __init__(self, model, preprocessor, device, action_selector=None, action_limits=None):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        if action_selector is None:
            action_selector = categorical_action_selector if model.is_discrete else normal_action_selector
        self.action_selector = action_selector
        self.action_limits = action_limits

    def __call__(self, state):
        in_ts = self.preprocessor.preprocess(state).to(self.device)

        with torch.no_grad():
            policy_out, vals_out = self.model(in_ts)
            actions = self.action_selector(policy_out, self.action_limits)

        return actions


class EpisodeResult(object):

    def __init__(self, env, start_state, episode_id=None, chain=True, partial_unroll=True):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.infos = []
        self.done = False
        self.episode_id = episode_id if episode_id is not None else str(uuid.uuid4())
        self.chain = chain
        self.get_offset = 0
        self.partial_unroll = partial_unroll
        self.next_episode_result = None

    def append(self, action, reward, state, done, info=None):
        if self.done:
            if not self.chain:
                raise ValueError("Can't append to done EpisodeResult.")
            else:
                self.next_episode_result.append(action, reward, state, done, info)
        else:
            self.actions.append(action)
            self.states.append(state)
            self.rewards.append(reward)
            self.done = done
            self.infos.append(info)

            if done and self.chain:
                self.begin_new_episode()

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return

    def n_step_return(self, n, gamma, last_state_value):
        cur_state, action, rewards = self.n_step_stats(n)
        result = 0.0 if self.done else last_state_value
        for r in reversed(rewards):
            result = r + gamma * result
        return result

    def n_step_idx(self, n):
        if self.chain and self.done and self.partial_unroll:
            idx = n - self.get_offset
        else:
            idx = n
        return -min(idx, len(self.rewards))

    def n_step_stats(self, n):
        n_step_idx = self.n_step_idx(n)
        cur_state = self.cur_state(n)
        rewards = self.rewards[n_step_idx:]
        action = self.actions[n_step_idx]
        return cur_state, action, rewards

    def cur_state(self, n):
        return self.states[max(self.n_step_idx(n) - 1, -len(self.states))]

    def cur_action(self, n):
        return self.actions[self.n_step_idx(n)]

    def get_final_return(self, gamma=1.0):
        if not self.done:
            return
        return self.calculate_return(gamma)

    def update_state(self, n, gamma=1.0):
        if not self.done:
            return

        if self.partial_unroll:
            self.get_offset += 1
        if self.get_offset >= n or not self.partial_unroll:
            final_return = self.get_final_return(gamma)
            self.set_to_next_episode_result()
            return final_return

    def begin_new_episode(self, episode_id=None, chain=True):
        self.next_episode_result = EpisodeResult(self.env, self.env.reset(), episode_id=episode_id, chain=chain,
                                                 partial_unroll=self.partial_unroll)

    def set_to_next_episode_result(self):
        self.env = self.next_episode_result.env
        self.states = self.next_episode_result.states
        self.actions = self.next_episode_result.actions
        self.rewards = self.next_episode_result.rewards
        self.infos = self.next_episode_result.infos
        self.done = self.next_episode_result.done
        self.episode_id = self.next_episode_result.episode_id
        self.chain = self.next_episode_result.chain
        self.get_offset = self.next_episode_result.get_offset
        self.next_episode_result = self.next_episode_result.next_episode_result

    @property
    def last_state(self):
        return self.states[-1]

    def __str__(self):
        return f"{self.actions} - {self.rewards}"

    def __len__(self):
        return len(self.states)


class ActorCriticBatch(object):

    def __init__(self, states=None, actions=None, values=None, advantages=None):
        if states is None:
            states = []
        if actions is None:
            actions = []
        if values is None:
            values = []
        if advantages is None:
            advantages = []
        self.states = states
        self.actions = actions
        self.values = values
        self.advantages = advantages

    def append(self, state, action, value, advantage):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.advantages.append(advantage)

    def get_batch(self, batch_size):
        sub_batch = ActorCriticBatch(self.states[:batch_size], self.actions[:batch_size], self.values[:batch_size],
                                     self.advantages[:batch_size])
        self.states = self.states[batch_size:]
        self.actions = self.actions[batch_size:]
        self.values = self.values[batch_size:]
        self.advantages = self.advantages[batch_size:]
        return sub_batch

    def __len__(self):
        return len(self.states)


class EnvironmentsDataset(object):

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, n_steps, gamma, batch_size, preprocessor,
                 device, action_selector=None, epoch_length=None, partial_unroll=True, action_limits=None):
        self.envs = {idx: e for idx, e in enumerate(envs)}
        self.model = model
        self.num_actions = model.action_dimension
        self.discrete = model.is_discrete
        self.action_limits = action_limits
        if n_steps < 1:
            raise ValueError(f"Number of steps {n_steps} needs be greater or equal to 1")
        self.n_steps = n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.device = device
        if action_selector is None:
            action_selector = categorical_action_selector if model.is_discrete else normal_action_selector
        self.action_selector = action_selector
        self.episode_results = {}
        self.epoch_length = epoch_length
        self.partial_unroll = partial_unroll
        self.reset()

    def data(self):
        batch = ActorCriticBatch()
        er_returns = []
        cur_batch_idx = 0

        while True:
            sorted_ers = sorted(self.episode_results.items())
            k_to_idx = {k: idx for idx, (k, v) in enumerate(sorted_ers)}

            in_ts = torch.cat([self.preprocessor.preprocess(er.last_state) for k, er in sorted_ers]).to(self.device)

            with torch.no_grad():
                policy_out, vals_out = self.model(in_ts)

                actions = self.action_selector(policy_out, self.action_limits)

            self.step(actions)

            to_train_ers = {k: er for k, er in sorted_ers if (len(er) > self.n_steps)
                            or len(er) <= self.n_steps and er.done}

            if len(to_train_ers) > 0:
                last_states_vals = [float(vals_out[k_to_idx[k]]) for k in to_train_ers.keys()]
                batch_ers = [er for k, er in to_train_ers.items()]
                n_step_returns = [er.n_step_return(self.n_steps, self.gamma, l_v) for er, l_v in
                                  zip(batch_ers, last_states_vals)]

                with torch.no_grad():
                    cur_in_ts = torch.cat(
                        [self.preprocessor.preprocess(er.cur_state(self.n_steps)) for k, er in
                         to_train_ers.items()]).to(self.device)
                    _, cur_vals_out = self.model(cur_in_ts)

                advantages = [n_r - float(c_v) for n_r, c_v in zip(n_step_returns, cur_vals_out)]

                for er, val, adv in zip(batch_ers, n_step_returns, advantages):
                    batch.append(self.preprocessor.preprocess(er.cur_state(self.n_steps)), er.cur_action(self.n_steps),
                                 float(val), float(adv))

                er: EpisodeResult
                for er in batch_ers:
                    len_er = len(er)
                    er_r_ud = er.get_final_return()
                    er_r = er.update_state(self.n_steps, gamma=self.gamma)
                    if er_r is not None:
                        er_returns.append((len_er, er_r, er_r_ud))

                if len(batch) >= self.batch_size:
                    yield er_returns, batch.get_batch(self.batch_size)

                    er_returns = []

                    if self.epoch_length is not None:
                        cur_batch_idx += 1
                        if cur_batch_idx >= self.epoch_length:
                            return

    def reset(self):
        self.episode_results = {k: EpisodeResult(e, e.reset(), partial_unroll=self.partial_unroll) for k, e in
                                self.envs.items()}

    def step(self, actions):
        for (k, er), a in zip(sorted(self.episode_results.items()), actions):
            s, r, d, i = er.env.step(a)
            er.append(a, r, s, d, i)
