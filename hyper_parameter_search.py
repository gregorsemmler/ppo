import argparse
import logging
import os
from argparse import SUPPRESS
from copy import deepcopy
from datetime import datetime
from os.path import join
from types import SimpleNamespace

import numpy as np

from common import load_json, save_json
from train import TRAIN_ARG_PARSER, training, get_model_from_args


logger = logging.getLogger(__name__)


def log10_uniform(low=1e-5, high=1e-2):
    return np.power(10, np.random.uniform(np.log10(low), np.log10(high)))


def norm_type(obj):
    if isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj


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
                return lambda: norm_type(np.random.uniform(low, high))
            elif isinstance(low, int):
                return lambda: norm_type(np.random.randint(low, high))
        elif sample_type == "log10_uniform":
            low, high = sample_limits
            return lambda: norm_type(log10_uniform(low, high))
        elif sample_type == "choice":
            return lambda: norm_type(sample_limits[np.random.randint(len(sample_limits))])
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
    parser.add_argument("--save_path", type=str, default="hyperparam_search_results")
    parser.add_argument("--hyperparams_path", type=str, required=True)
    parser.add_argument("--run_id", default=None)
    search_args = parser.parse_args()
    n_rounds = search_args.n_rounds
    best_key = search_args.best_key
    run_id = search_args.run_id if search_args.run_id is not None else f"run_{datetime.now():%d%m%Y_%H%M%S}"

    namespace = SimpleNamespace()
    for action in TRAIN_ARG_PARSER._actions:
        if action.dest is not SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not SUPPRESS:
                    setattr(namespace, action.dest, action.default)

    hyperparams = load_json(search_args.hyperparams_path)
    save_path = search_args.save_path
    os.makedirs(save_path, exist_ok=True)

    tuner = RandomSearchTuner(hyperparams, n_rounds)
    search_results = []

    best_value = float("-inf")
    best_config = None
    best_metrics = None

    for config_id, config in enumerate(tuner.get_configurations()):
        logger.info(f"Evaluating config #{config_id}: {config}")
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
                best_metrics = {"config": config, "metrics": metrics}
                if best_metrics is not None:
                    save_json(join(save_path, f"{run_id}_best_metrics.json"), best_metrics)

    logger.info(f"Best config: {best_config}")
    logger.info(f"Best value: {best_value}")
    save_json(join(save_path, f"{run_id}_search_results.json"), search_results)
    print("")


if __name__ == "__main__":
    search_parameters()
