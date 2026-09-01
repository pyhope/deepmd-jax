import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

setup(
    name='deepmd_jax',
    version='0.2.1',
    packages=find_packages(),
    install_requires=[
        'jax[cuda12]>=0.7.1,<0.8',
        'flax>=0.11,<0.12',
        'optax>=0.2.5,<0.3',
        'jax-md @ https://github.com/google/jax-md/archive/a41c7d19f6468f4e5263c32c12c9ed6cba26ebff.tar.gz',
        'ase',
        'matplotlib',
        'gpustat',
        'ipykernel',
    ],
    author='Ruiqi Gao',
    author_email='ruiqigao@princeton.edu',
    description='DP-JAX',
    url='https://github.com/SparkyTruck/deepmd-jax',
    python_requires='>=3.10',
)
