# CarEnv

Easy to use `gym`-Environment for 2D vehicle simulations using dynamic scenes.

## Racing scenario

Navigating a randomly generated tightly winding road at speed. The simulated vehicle uses a dynamic single track model with a coupled Dugoff tire model.
Throttle, brake and steering are continuous actions, with the vehicle by default using front wheel drive.
The agent may learn to control brake balance by applying throttle and brake individually.

![Video of agent on country road environment](Docs/CountryRoadShort.gif)

## Parking scenario

Parallel parking in reverse using a kinematic model. Steering and acceleration (positive through negative) 
are continuous actoins.

![Video of agent on parking environment](Docs/ParkingShort.gif)

## Installation

To install the latest version, simply run:
```shell
pip install git+https://github.com/m-schier/CarEnv
```

You may then create a new gym environment, e.g. on the `racing` configuration:
```python
from CarEnv import CarEnv, Configs

env = CarEnv(Configs.get_standard_env_config("racing"))
```

However, if you want to modify the environment or run any of our example scripts it may be more convenient to clone this repository and then install using local linking:
```shell
pip install -e .
```

## Running as human
Execute `scripts/run_human.py`. The agent may be controlled by keyboard or by a joystick or
steering wheel if present. You may have to modify the axis and button numbers when using a controller,
see the implementation in `CarEnv/Actions/` for available keyword arguments.

## Training a Soft Actor-Critic agent
In `scripts/train_sac.py` you may find an example script on how to train a Soft Actor-Critic
using a Deep Set feature extractor on the `parking` and `racing` configurations. This
implementation uses the Stable Baselines 3 library. You must install the reinforcement learning extra requirements, i.e.:
```shell
pip install -e .[RL]
```

## Citing
If you find this environment useful, you may cite it by our paper in which it was initially presented:

```bibtex
@inproceedings { SchRei2023b,
  author = {Maximilian Schier and Christoph Reinders and Bodo Rosenhahn},
  title = {Learned Fourier Bases for Deep Set Feature Extractors in Automotive Reinforcement Learning},
  booktitle = {2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
  year = {2023},
  month = sep,
  pages = {to appear}
}
