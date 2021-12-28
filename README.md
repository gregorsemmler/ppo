## Proximal Policy Optimization (PPO)

A clean and modular implementation of Proximal Policy Optimization as described in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) written in PyTorch. 

Implemented and tested with PyTorch 1.9.0, Gym 0.15.4, Roboschool 1.0.48, PyBullet 3.2.0 and OpenCV 4.5.4.58 on Ubuntu 20.04.

* `atari_wrappers.py` contains wrappers for gym atari environments.
* `common.py` contains some common functionality.
* `data.py` contains code for the dataset.
* `envs.py` contains a simple test environment.
* `model.py` contains multiple model definitions.
* `play.py` allows testing and visualizing previously trained models.
* `train.py` contains the main training code with the opportunity to customize hyperparameters, the environment being trained, the model and the ability to save and load models.
* `hyper_parameter_search.py` contains code to perform multiple training runs and evaluate which hyperparameters perform the best based on a defined metric. 

Example command for training:

`python3 train.py --env_name RoboschoolHalfCheetah-v1 --run_id HalfCheetah-Run1 --checkpoint_path model_checkpoints --entropy_factor 0 --batch_size 128 --lr 1e-4 --critic_lr 1e-3 --gamma 0.99 --lambd 0.95 --n_steps 2100 --tensorboardlog`

Example command for playing:

`python3 play.py --env_name RoboschoolAnt-v1 --fixed_std --model_path model_checkpoints/best/Ant-v1_123.tar --video_path videos/Ant-v1 --render`

Example command for hyperparameter search:

`python3 hyper_parameter_search.py --best_key eval_epoch/undisc_return --hyperparams_path example_hyperparams.json --n_processes 3 --n_rounds 50 --run_id AntParameterSearch1`

&nbsp;

![Ant Example](example_gifs/Ant.gif)
![BipedalWalker Example](example_gifs/BipedalWalker.gif)
![HalfCheetah Example](example_gifs/HalfCheetah.gif)
![LunarLander Example](example_gifs/LunarLander.gif)
![Reacher Example](example_gifs/Reacher.gif)
