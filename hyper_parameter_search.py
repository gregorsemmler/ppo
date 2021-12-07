import argparse
import logging
from argparse import SUPPRESS
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from common import load_json
from train import TRAIN_ARG_PARSER, training, get_model_from_args


def log10_uniform(low=1e-5, high=1e-2):
    return np.power(10, np.random.uniform(np.log10(low), np.log10(high)))


class RandomSearchTuner(object):

    def __init__(self, hyper_param_dict, n_rounds=None):
        self.hyperparams = {}
        for k, val in hyper_param_dict.items():
            self.hyperparams[k] = self.parse_hyper_param_entry(val) if isinstance(val, list) else val
        self.n_rounds = n_rounds

    @staticmethod
    def parse_hyper_param_entry(value):
        sample_type, sample_limits = value

        if sample_type == "uniform":
            low, high = sample_limits
            if isinstance(low, float):
                return lambda: float(np.random.uniform(low, high))
            elif isinstance(low, int):
                return lambda: int(np.random.randint(low, high))
        elif sample_type == "log10_uniform":
            low, high = sample_limits
            return lambda: float(log10_uniform(low, high))
        elif sample_type == "choice":
            return lambda: np.random.choice(sample_limits)
        else:
            raise ValueError(f"Unknown Sample Type {sample_type}")

    def get_configurations(self):
        if self.n_rounds is not None:
            for _ in range(self.n_rounds):
                yield {k: v() if callable(v) else v for k, v in self.hyperparams.items()}
        else:
            while True:
                yield {k: v() if callable(v) else v for k, v in self.hyperparams.items()}


def search_parameters():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--best_key", type=str)
    search_args = parser.parse_args()
    n_rounds = search_args.n_rounds
    best_key = search_args.best_key

    namespace = SimpleNamespace()
    for action in TRAIN_ARG_PARSER._actions:
        if action.dest is not SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not SUPPRESS:
                    setattr(namespace, action.dest, action.default)

    hyperparams_path = "example_hyperparams.json"
    hyperparams = load_json(hyperparams_path)

    tuner = RandomSearchTuner(hyperparams, n_rounds)
    search_results = []

    best_value = float("-inf")
    best_config = None

    for config in tuner.get_configurations():
        new_args = deepcopy(namespace)
        for attr_name, value in config.items():
            setattr(new_args, attr_name, value)

        model, device = get_model_from_args(new_args)
        metrics = training(new_args, model, device)
        search_results.append((config, metrics))

        if best_key is not None:
            cur_val = np.mean(metrics[best_key])
            if cur_val > best_value:
                best_value = cur_val
                best_config = config
        print("")

    print("")
    pass


if __name__ == "__main__":
    search_parameters()
