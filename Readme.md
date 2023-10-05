# CarEnv

Easy to use `gym`-Environment for 2D vehicle simulations using dynamic scenes.

## Racing scenario

Navigating a tightly winding road with slippery surface. Vehicle uses a dynamic single track model with
front wheel drive.
Throttle, brake and steering are continuous actions. The agent may learn to control brake
balance by applying throttle and brake individually.

![Video of agent on country road environment](Docs/CountryRoadShort.gif)

## Parking scenario

Parallel parking in reverse using a kinematic model. Steering and acceleration (positive through negative) 
are continuous actoins.

![Video of agent on parking environment](Docs/ParkingShort.gif)

## Installation

Run from the command prompt using the python environment for development from this folder:
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
