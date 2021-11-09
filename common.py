import logging
import signal

import gym
from gym.spaces import Discrete, Box

import torch

from atari_wrappers import wrap_deepmind, make_atari
from envs import SimpleCorridorEnv
from model import SharedMLPModel, SimpleCNNPreProcessor, CNNModel, MLPModel, NoopPreProcessor

logger = logging.getLogger(__name__)


CHECKPOINT_MODEL = "model"
CHECKPOINT_OPTIMIZER = "optimizer"
CHECKPOINT_CRITIC_OPTIMIZER = "critic_optimizer"
CHECKPOINT_MODEL_ID = "model_id"


def load_checkpoint(path, model, optimizer=None, critic_optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state[CHECKPOINT_MODEL])
    if optimizer is not None:
        optimizer.load_state_dict(state[CHECKPOINT_OPTIMIZER])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(state[CHECKPOINT_CRITIC_OPTIMIZER])
    epoch = state.get(CHECKPOINT_MODEL_ID)
    return epoch


def save_checkpoint(path, model, optimizer=None, critic_optimizer=None, model_id=None):
    torch.save({
        CHECKPOINT_MODEL: model.state_dict(),
        CHECKPOINT_OPTIMIZER: optimizer.state_dict() if optimizer is not None else None,
        CHECKPOINT_CRITIC_OPTIMIZER: critic_optimizer.state_dict() if critic_optimizer is not None else None,
        CHECKPOINT_MODEL_ID: model_id,
    }, path)


def get_action_space_details(action_space):
    if isinstance(action_space, Discrete):
        discrete = True
    elif isinstance(action_space, Box):
        discrete = False
    else:
        raise ValueError("Unknown type of action_space")

    action_dim = action_space.n if discrete else action_space.shape[0]
    limits = None if discrete else (float(action_space.low), float(action_space.high))
    return discrete, action_dim, limits


def get_model(env_name, shared_model, atari, device, fixed_std=True):
    if env_name == "SimpleCorridor":
        eval_env = SimpleCorridorEnv()
        state = eval_env.reset()
        in_states = state.shape[0]
        discrete, action_dim, limits = get_action_space_details(eval_env.action_space)
        if shared_model:
            return SharedMLPModel(in_states, action_dim, fixed_std=fixed_std, discrete=discrete).to(device)
        return MLPModel(in_states, action_dim, fixed_std=fixed_std, discrete=discrete).to(device)
    elif atari:
        eval_env = wrap_deepmind(make_atari(env_name))
        state = eval_env.reset()

        preprocessor = SimpleCNNPreProcessor()
        in_t = preprocessor.preprocess(state)
        discrete, action_dim, limits = get_action_space_details(eval_env.action_space)
        input_shape = tuple(in_t.shape)[1:]
        return CNNModel(input_shape, action_dim, discrete=discrete, fixed_std=fixed_std).to(device)

    eval_env = gym.make(env_name)
    state = eval_env.reset()
    in_states = state.shape[0]
    discrete, action_dim, limits = get_action_space_details(eval_env.action_space)
    if shared_model:
        return SharedMLPModel(in_states, action_dim, fixed_std=fixed_std, discrete=discrete).to(device)
    return MLPModel(in_states, action_dim, fixed_std=fixed_std, discrete=discrete).to(device)


def get_preprocessor(env_name, atari):
    if env_name == "SimpleCorridor":
        return NoopPreProcessor()
    elif atari:
        return SimpleCNNPreProcessor()
    return NoopPreProcessor()


def get_environment(env_name, atari):
    if env_name == "SimpleCorridor":
        return SimpleCorridorEnv()
    elif atari:
        return wrap_deepmind(make_atari(env_name))
    return gym.make(env_name)


class GracefulExit(object):

    def __init__(self):
        self.run = True
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        logger.info("Termination Signal received. Exiting gracefully")
        self.run = False
