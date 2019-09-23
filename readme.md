# Overview
RLtime is a reinforcement learning library, currently supporting PyTorch, with focus on state-of-the-art q-learning algorithms and features, and interacting with real-time environments which require low-latency acting and sample-efficient training.


# Supported Policies and Features
Following policies are currently supported:
- DQN ([Deep Q Networks](https://arxiv.org/abs/1312.5602))
- IQN ([Implicit Quantile Networks](https://arxiv.org/abs/1806.06923))
- [Distributional DQN / C51](https://arxiv.org/abs/1707.06887) (Unverified)
- Actor Critic (Trainable via [A2C](https://arxiv.org/abs/1602.01783) or [PPO](https://arxiv.org/abs/1707.06347))

All q-learning algorithms commonly support any combination of most state-of-the-art
features, in particular:
- All [rainbow](https://arxiv.org/abs/1710.02298) features (except noisy nets) are supported:
    - Double Q-Learning
    - Dueling Networks
    - Multi-Step Targets
    - Prioritized Replay
- [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) features:
    - Value Function Rescaling (Instead of reward clipping)
    - Weighted multi-step/multi-env prioritized replay (With configurable sequence overlap)
    - Stored recurrent state
    - Recurrent state burn-in

All algorithms support recurrent models with dynamic model layout configuration using json files, with [extended syntax](https://github.com/opherlieber/rltime/blob/master/docs/json_syntax.md) for referencing/embedding other json files and python types.


# Atari Results
By extending IQN with an LSTM model as well as some rainbow and R2D2 features
we can achieve state-of-the-art (in sample efficiency, 200M frames) results on
the Atari benchmark:

|Algo|Median Normalized Score|
|----|----|
|Impala Deep|192%|
|QR-DQN-1|211%|
|IQN|218%|
|Rainbow|230%|
|Recurrent IQN|365%<sup>*</sup>|

<sup>* For 55 out of 57 Atari-57 games, single run per game</sup>

A write-up with additional details and full results can be found [here](https://opherlieber.github.io/rl/2019/09/22/recurrent_iqn).

# Installation
```bash
git clone https://github.com/opherlieber/rltime.git
cd rltime
pip install -e .

# For tensorboard logging (Requires PyTorch 1.1+ and tensorboard should be at least version 1.14)
pip install tensorboard Pillow
```
For distributed acting using Ray see the [ray docs](https://ray.readthedocs.io/en/latest/installation.html) for ray installation and cluster setup instructions, as well as additional details [here](https://github.com/opherlieber/rltime/blob/master/docs/distributed_acting.md).



# Usage
```bash
# Training a config
python -um rltime.train cartpole_ppo.json
# Training a config with a specific ENV
python -um rltime.train atari_iqn_lstm.json --env BreakoutNoFrameskip-v4

# Evaluate a training result on 32 parallel ENVs for 100 total episodes
# (Result is logged to 'eval.json' in <result_dir>)
python -um rltime.eval <result_dir> --num-envs 32 --episodes 100

# Evaluate a training result and render to the screen in real-time, 4 ENVs tiled
python -um rltime.eval <result_dir> --num-envs 4 --episodes 10 --render
```

# Acting/Training
The library implements a full separation between acting and training. Any
acting configuration can be run with any policy/training configuration.

The following acting modes are supported:
- Vectorized multi-env local/synchronized acting using the training policy directly
- High-throughput async/distributed acting (Using Ray, still experimental and not fully verified)

# History Buffers
The library has a full separation between acting/training and the history buffer used to accumulate acting data and generate training batches. In particular we can run any training algorithm using any history buffer:
- Online History (E.g. for actor-critic algos, or online nstep q-learning using any of the q-learning algos)
- Replay and Prioritized Replay (Including support for weighted multi-env/multi-step prioritized sequence replay)

# Backend
RLtime currently only supports pytorch as a backend (1.0+, 1.1+ recommended), though there is some preparation to possibly add support for TF2.0 later and commonize some of the code.

In general, any code outside of a 'torch' subdirectory should currently be backend independent.

# Citing the Project
Please cite this repository if using RLtime in your research or work:

```
@misc{rltime,
  author = {Lieber, Opher},
  title = {RLtime: A reinforcement learning library for state-of-the-art q-learning},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/opherlieber/rltime}},
}
```