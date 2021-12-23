import argparse
import logging
import os
import queue
from argparse import SUPPRESS
from copy import deepcopy
from datetime import datetime
from os.path import join
from types import SimpleNamespace

import numpy as np
import torch.multiprocessing as mp

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


def evaluate_configs(namespace, configs, best_key, run_id, save_path, trainer_id=0):
    search_results = []

    best_value = float("-inf")
    best_config = None
    best_metrics = None

    config_id = 0
    for config in configs:
        logger.info(f"{trainer_id}# Evaluating config #{config_id}: {config}")
        new_args = deepcopy(namespace)

        for attr_name, value in config.items():
            setattr(new_args, attr_name, value)

        model, device = get_model_from_args(new_args)

        try:
            metrics = training(new_args, model, device, trainer_id=trainer_id)
        except Exception as e:
            logger.exception("Encountered unexpected error turning training.")
            logger.warning("Continuing to next configuration.")
            continue

        cur_val = np.mean(metrics[best_key])
        best_metrics = {"config": config, "key": best_key, "value": cur_val, "metrics": metrics}
        search_results.append(best_metrics)

        if cur_val > best_value:
            best_value = cur_val
            best_config = config
            best_metrics = {"config": config, "key": best_key, "value": best_value, "metrics": metrics}
            if best_metrics is not None:
                save_json(join(save_path, f"{run_id}_best_metrics.json"), best_metrics)

        config_id += 1

    logger.info(f"Best config: {best_config}")
    logger.info(f"Best value: {best_value}")
    save_json(join(save_path, f"{run_id}_search_results.json"), search_results)

    return best_metrics, search_results


def search_parameters():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, default=100)
    parser.add_argument("--n_processes", type=int, default=1)
    parser.add_argument("--best_key", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="hyperparam_search_results")
    parser.add_argument("--hyperparams_path", type=str, required=True)
    parser.add_argument("--run_id", default=None)
    search_args = parser.parse_args()

    n_processes = search_args.n_processes
    n_rounds = search_args.n_rounds
    best_key = search_args.best_key
    run_id = search_args.run_id if search_args.run_id is not None else f"run_{datetime.now():%d%m%Y_%H%M%S}"

    namespace = SimpleNamespace()
    known_params = set()
    for action in TRAIN_ARG_PARSER._actions:
        if action.dest is not SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not SUPPRESS:
                    known_params.add(action.dest)
                    setattr(namespace, action.dest, action.default)

    hyperparams = load_json(search_args.hyperparams_path)

    unknown_params = hyperparams.keys() - known_params
    if len(unknown_params) > 0:
        raise RuntimeError(f"Unknown Parameters: {sorted(unknown_params)}")

    save_path = search_args.save_path
    os.makedirs(save_path, exist_ok=True)

    tuner = RandomSearchTuner(hyperparams, n_rounds)

    if n_processes == 1:
        evaluate_configs(namespace, tuner.get_configurations(), best_key, run_id, save_path)
    else:
        processes = []
        input_queue = mp.Queue()
        return_queue = mp.Queue()

        for c in tuner.get_configurations():
            input_queue.put(c)

        for proc_idx in range(n_processes):
            p = mp.Process(target=multiprocess_wrapper,
                           args=(proc_idx, namespace, best_key, run_id, save_path, input_queue, return_queue))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        best_overall_metrics = None
        best_overall_value = float("-inf")
        overall_search_results = []
        while return_queue.qsize():
            proc_ret = return_queue.get()
            proc_best_metrics, proc_search_results = proc_ret
            proc_best_val = proc_best_metrics["value"]
            if proc_best_val > best_overall_value:
                best_overall_value = proc_best_val
                best_overall_metrics = proc_best_metrics

            overall_search_results.extend(proc_search_results)

        if best_overall_metrics is not None:
            logger.info(f"Best overall config: {best_overall_metrics['config']}")
            logger.info(f"Best overall value: {best_overall_value}")

            overall_bm_save_path = join(save_path, f"{run_id}_best_metrics.json")
            overall_sr_save_path = join(save_path, f"{run_id}_search_results.json")
            save_json(overall_bm_save_path, best_overall_metrics)
            save_json(overall_sr_save_path, overall_search_results)

            logger.info(f"Saved overall best metrics to '{overall_bm_save_path}'")
            logger.info(f"Saved overall search results to '{overall_sr_save_path}'")


def get_multiprocess_configs(input_queue: mp.Queue, timeout=0.01):
    while True:
        try:
            config = input_queue.get(timeout=timeout)
            yield config
        except queue.Empty:
            break


def multiprocess_wrapper(process_idx, namespace, best_key, run_id, save_path, input_queue: mp.Queue,
                         return_queue: mp.Queue):
    proc_run_id = f"{run_id}_{process_idx}"
    return_queue.put(
        evaluate_configs(namespace, get_multiprocess_configs(input_queue), best_key, proc_run_id, save_path,
                         trainer_id=process_idx))


if __name__ == "__main__":
    search_parameters()
