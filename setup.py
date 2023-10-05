from setuptools import setup, find_packages

setup(
    name='CarEnv',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium~=0.29.1',
        'matplotlib>=3.5',
        'numba>=0.56',
        'numpy>=1.22',
        'pycairo~=1.21',
        'pygame~=2.1',
        'Shapely~=1.8.4'
    ],
    extras_require={
        'RL': ['rich==13.4.2', 'stable-baselines3==2.1.0', 'tensorboard==2.10.0', 'tqdm==4.64.0', 'wandb==0.13.2'],
    },
)
