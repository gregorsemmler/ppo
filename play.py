import argparse
import logging
import os
from datetime import datetime
from os.path import join

import roboschool
import torch
from gym.wrappers import Monitor

from common import load_checkpoint, get_environment
from data import Policy, EpisodeResult
from model import get_model, get_preprocessor

logger = logging.getLogger(__name__)


def play_environment(env, policy, num_episodes=1, render=False, gamma=1.0, video_save_path=None, run_id=None,
                     verbose=True):
    i = 0
    best_return = float("-inf")
    best_result = None
    episode_infos = []

    if video_save_path is not None:
        sub_folder = f"{datetime.now():%d%m%Y_%H%M%S}" if run_id is None else str(run_id)
        sub_path = join(video_save_path, sub_folder)
        os.makedirs(sub_path, exist_ok=True)
        env = Monitor(env, sub_path, video_callable=lambda x: True, force=True)

    while i < num_episodes:
        state = env.reset()
        done = False

        episode_result = EpisodeResult(env, state)
        while not done:
            if render:
                env.render()

            action = policy(state).squeeze()
            new_state, reward, done, info = env.step(action)

            episode_result.append(action, reward, new_state, done, info)

            state = new_state

        episode_return = episode_result.calculate_return(gamma)
        undisc_return = episode_result.calculate_return(1.0)
        ep_len = len(episode_result)
        if best_return < episode_return:
            best_return = episode_return
            best_result = episode_result
            if verbose:
                logger.info(f"New best return: {best_return:.5g}")

        episode_infos.append((episode_return, undisc_return, ep_len))
        if verbose:
            logger.info(f"Episode {i}: Return: {episode_return:.5g} "
                        f"(Undiscounted: {undisc_return:.5g}) Length: {ep_len}")
        i += 1

    return episode_infos, best_result, best_return


def evaluate_model():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--device_token", default=None)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--atari", dest="atari", action="store_true")
    parser.add_argument("--no_atari", dest="atari", action="store_false")
    parser.add_argument("--shared_model", dest="shared_model", action="store_true")
    parser.add_argument("--no_shared_model", dest="shared_model", action="store_false")
    parser.add_argument("--fixed_std", dest="fixed_std", action="store_true")
    parser.add_argument("--no_fixed_std", dest="fixed_std", action="store_false")
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.set_defaults(atari=False, shared_model=False, fixed_std=False, render=True)
    args = parser.parse_args()

    env_name = args.env_name
    atari = args.atari
    shared_model = args.shared_model
    model_path = args.model_path
    gamma = args.gamma
    render = args.render
    video_path = args.video_path
    num_episodes = args.n_episodes
    run_id = args.run_id

    if args.device_token is None:
        device_token = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_token = args.device_token

    device = torch.device(device_token)

    model = get_model(env_name, shared_model, atari, device, fixed_std=args.fixed_std)
    preprocessor = get_preprocessor(env_name, atari)
    env = get_environment(env_name, atari)

    load_checkpoint(model_path, model, device=device)

    policy = Policy(model, preprocessor, device)

    play_environment(env, policy, num_episodes=num_episodes, render=render, gamma=gamma, video_save_path=video_path,
                     run_id=run_id)


if __name__ == "__main__":
    evaluate_model()
