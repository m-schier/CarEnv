from setuptools import setup, find_packages

setup(
    name='CarEnv',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'gym==0.21.0',  # Fix to this version because broken backwards compatibility afterwards
        'matplotlib>=3.5',
        'numba>=0.56',
        'numpy>=1.22',
        'pycairo~=1.21',
        'pygame~=2.1',
        'Shapely~=1.8.4'
    ],
)
