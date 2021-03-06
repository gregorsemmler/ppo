import logging
import uuid
from typing import Sequence, Dict

import numpy as np
import torch
from gym import Env

from model import ActorCriticModel

logger = logging.getLogger(__name__)


class Policy(object):

    def __init__(self, model, preprocessor, device, action_limits=None, squeeze_output=True):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.action_limits = action_limits
        self.squeeze_output = squeeze_output

    def __call__(self, state):
        in_ts = self.preprocessor.preprocess(state).to(self.device)

        with torch.no_grad():
            policy_out, vals_out = self.model(in_ts)
            actions = self.model.select_actions(policy_out, self.action_limits)

        if self.squeeze_output:
            actions = actions.squeeze()

        return actions


class EpisodesBuffer(object):

    def __init__(self, start_state):
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.infos = []

    def append(self, action, reward, state, done, value, log_prob, info=None):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
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

    def reset(self, start_state):
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.infos = []


class EpisodeResult(object):

    def __init__(self, env, start_state, episode_id=None):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []
        self.infos = []
        self.done = False
        self.episode_id = episode_id if episode_id is not None else str(uuid.uuid4())
        self.get_offset = 0

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

    def __len__(self):
        return len(self.states)


class PPOBatch(object):

    def __init__(self, states, actions, values, advantages, log_probs):
        self.states = states
        self.actions = actions
        self.values = values
        self.advantages = advantages
        self.log_probs = log_probs

    def __len__(self):
        return len(self.states)


class EnvironmentsDataset(object):

    def __init__(self, envs: Sequence[Env], model: ActorCriticModel, n_steps, gamma, lambd, num_ppo_rounds, batch_size,
                 preprocessor, device, action_limits=None):
        self.envs: Dict[int, Env] = {idx: e for idx, e in enumerate(envs)}
        self.model = model
        self.num_actions = model.action_dimension
        self.discrete = model.is_discrete
        self.action_limits = action_limits
        if n_steps < 1:
            raise ValueError(f"Number of steps {n_steps} needs be greater or equal to 1")
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambd = lambd
        self.num_ppo_rounds = num_ppo_rounds
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.device = device
        self.episode_buffers = {}

    def calculate_gae_and_value(self, eps_buffer: EpisodesBuffer):
        gae = 0.0
        gaes = []
        values = []
        episode_infos = []
        cur_return = 0.0
        cur_undiscounted_return = 0.0
        cur_ep_len = 0
        complete_final_episode = eps_buffer.dones[-2]

        for idx, (state, action, reward, done, value, next_value, info) in enumerate(list(reversed(eps_buffer))[1:]):
            if done:
                delta = reward - value
                gae = delta
                if cur_ep_len > 0:
                    episode_infos.append((cur_ep_len, cur_return, cur_undiscounted_return))
                    cur_return = 0.0
                    cur_undiscounted_return = 0.0
                    cur_ep_len = 0
            else:
                delta = reward + self.gamma * next_value - value
                gae = delta + self.gamma * self.lambd * gae

            cur_return = self.gamma * cur_return + reward
            cur_undiscounted_return += reward
            cur_ep_len += 1

            gaes.append(gae)
            values.append(gae + value)

        if cur_ep_len > 0:
            episode_infos.append((cur_ep_len, cur_return, cur_undiscounted_return))

        if not complete_final_episode:
            episode_infos = episode_infos[1:]

        advantage_t = torch.FloatTensor(list(reversed(gaes)))
        value_t = torch.FloatTensor(list(reversed(values)))
        return advantage_t, value_t, episode_infos

    def data(self):
        self.reset()

        while True:

            in_ts = torch.cat([self.preprocessor.preprocess(eb.last_state)
                               for k, eb in self.episode_buffers]).to(self.device)

            with torch.no_grad():
                policy_out, vals_out = self.model(in_ts)

                actions = self.model.select_actions(policy_out, self.action_limits)
                log_probs = self.model.log_prob(policy_out, actions).detach().cpu()
                vals_out = vals_out.detach().cpu()

            eb: EpisodesBuffer
            for (k, eb), action, value, logprob in zip(self.episode_buffers, actions, vals_out, log_probs):
                state, reward, done, info = self.envs[k].step(action)

                if done:
                    state = self.envs[k].reset()

                eb.append(action, reward, state, done, value, logprob, info)

            if len([(k, eb) for k, eb in self.episode_buffers if (len(eb) > self.n_steps)]) == len(self.envs):
                advs, vs, ep_returns, ep_lengths = [], [], [], []

                for k, eb in self.episode_buffers:
                    a_t, v_t, ep_rs = self.calculate_gae_and_value(eb)
                    advs.append(a_t)
                    vs.append(v_t)
                    ep_returns.extend(ep_rs)

                adv_t = torch.cat(advs)
                val_t = torch.cat(vs)

                adv_std, adv_mean = torch.std_mean(adv_t)
                adv_t = (adv_t - adv_mean) / adv_std

                states_t = [self.preprocessor.preprocess(s) for k, eb in self.episode_buffers for s in
                            eb.states[:len(adv_t)]]
                states_t = torch.cat(states_t)
                actions_t = torch.from_numpy(np.stack(
                    [a for k, eb in self.episode_buffers for a in eb.actions[:-1]]))
                log_prob_t = torch.stack([lp for k, eb in self.episode_buffers for lp in eb.log_probs[:-1]])
                if len(actions_t.shape) == 1:
                    actions_t = actions_t[:, np.newaxis]
                if len(log_prob_t.shape) == 1:
                    log_prob_t = log_prob_t[:, np.newaxis]
                if len(adv_t.shape) == 1:
                    adv_t = adv_t[:, np.newaxis]

                assert len(states_t) == len(actions_t) == len(adv_t) == len(val_t) == len(log_prob_t) == self.n_steps

                for _ in range(self.num_ppo_rounds):
                    for batch_offset in range(0, self.n_steps, self.batch_size):
                        batch_states_t = states_t[batch_offset: batch_offset + self.batch_size]
                        batch_actions_t = actions_t[batch_offset: batch_offset + self.batch_size]
                        batch_adv_t = adv_t[batch_offset: batch_offset + self.batch_size]
                        batch_val_t = val_t[batch_offset: batch_offset + self.batch_size]
                        batch_log_prob_t = log_prob_t[batch_offset: batch_offset + self.batch_size]
                        batch = PPOBatch(batch_states_t, batch_actions_t, batch_val_t, batch_adv_t, batch_log_prob_t)
                        yield ep_returns, batch

                return

    def reset(self):
        self.episode_buffers = sorted([(k, EpisodesBuffer(e.reset())) for k, e in self.envs.items()])
