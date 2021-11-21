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


class EpisodesBuffer(object):

    def __init__(self, start_state):
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.infos = []

    def append(self, action, reward, state, done, value, info=None):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.infos.append(info)

    @property
    def last_state(self):
        return self.states[-1]

    @property
    def next_values(self):
        return self.values[1:] + [None]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx], \
               self.values[idx], self.next_values[idx], self.infos[idx]


# TODO refactor
class EpisodeResultOld(object):

    def __init__(self, env, start_state, episode_id=None, partial_unroll=True):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.infos = []
        self.done = False
        self.episode_id = episode_id if episode_id is not None else str(uuid.uuid4())
        self.get_offset = 0
        self.partial_unroll = partial_unroll

    def append(self, action, reward, state, done, info=None):
        if self.done:
            raise ValueError("Can't append to done EpisodeResult.")
        else:
            self.actions.append(action)
            self.states.append(state)
            self.rewards.append(reward)
            self.done = done
            self.infos.append(info)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return

    def __str__(self):
        return f"{self.actions} - {self.rewards}"


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

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, n_steps, gamma, lambd, batch_size, preprocessor,
                 device, action_selector=None, epoch_length=None, action_limits=None):
        self.envs = {idx: e for idx, e in enumerate(envs)}
        self.model = model
        self.num_actions = model.action_dimension
        self.discrete = model.is_discrete
        self.action_limits = action_limits
        if n_steps < 1:
            raise ValueError(f"Number of steps {n_steps} needs be greater or equal to 1")
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambd = lambd
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.device = device
        if action_selector is None:
            action_selector = categorical_action_selector if model.is_discrete else normal_action_selector
        self.action_selector = action_selector
        self.episode_buffers = {}
        self.epoch_length = epoch_length
        self.reset()

    def calculate_gae_and_value(self, episodes_buffer: EpisodesBuffer):
        gae = 0.0
        gaes = []
        values = []

        for state, action, reward, done, value, next_value, info in reversed(episodes_buffer)[:-1]:
            if done:
                delta = reward - value
                gae = delta
            else:
                delta = reward + self.gamma * next_value - value
                gae = delta + self.gamma * self.lambd * gae

            gaes.append(gae)

        advantage_t = torch.FloatTensor(np.array(list(reversed(gaes)))).to(self.device)
        value_t = torch.FloatTensor(np.array(list(reversed(values)))).to(self.device)
        return advantage_t, value_t

    def data(self):
        batch = ActorCriticBatch()
        er_returns = []
        cur_batch_idx = 0

        while True:

            in_ts = torch.cat([self.preprocessor.preprocess(eb.last_state)
                               for k, eb in self.episode_buffers]).to(self.device)

            with torch.no_grad():
                policy_out, vals_out = self.model(in_ts)

                actions = self.action_selector(policy_out, self.action_limits)

            # TODO test
            eb: EpisodesBuffer
            for (k, eb), a, v in zip(self.episode_buffers, actions, vals_out):
                s, r, d, i = self.envs[k].step(a)
                eb.append(a, r, s, d, v, i)

            if len([(k, eb) for k, eb in self.episode_buffers if (len(eb) > self.n_steps)]) == len(self.envs):

                adv_v_per_env = [self.calculate_gae_and_value(eb) for k, eb in self.episode_buffers]
                adv_t, val_t = list(zip(*adv_v_per_env))
                adv_t = torch.cat(adv_t)
                val_t = torch.cat(val_t)

                adv_std, adv_mean = torch.std_mean(adv_t)
                adv_t = (adv_t - adv_mean) / adv_std

                states_t = [[self.preprocessor.preprocess(s) for s in eb.states[:-1]] for k, eb in self.episode_buffers]
                states_t = torch.cat(states_t)
                actions_t = torch.cat([a for k, eb in self.episode_buffers for a in eb.actions[:-1]])

                # TODO
                for eb, val, adv in zip(batch_ers, n_step_returns, advantages):
                    batch.append(self.preprocessor.preprocess(eb.cur_state(self.n_steps)), eb.cur_action(self.n_steps),
                                 float(val), float(adv))

                # TODO
                eb: EpisodesBuffer
                for eb in batch_ers:
                    len_er = len(eb)
                    er_r_ud = eb.get_final_return()
                    er_r = eb.update_state(self.n_steps, gamma=self.gamma)
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
        # TODO implement
        self.episode_buffers = sorted({k: EpisodesBuffer(e.reset()) for k, e in self.envs.items()})

