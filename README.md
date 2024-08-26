# Multi-Particle Cryo-EM

This package is intended to function as the repository for data wrangling and analysis.

This package uses either CPUs or GPUs (using the [`cupy`](https://cupy.dev/) package).
To use the GPU-enabled version, see the GPU install section below

Additionally, the package organization assumes that a POSIX compliant system is present *(Linux or MacOS)*. Ideally you should be running this in a Linux environment.

## Pre-Installation

These are directions on how to get your python environment ready for installation

#### Pyenv pre-installation

To get ready for installing the first time using [venv](https://docs.python.org/3/library/venv.html):
```
git git@code.ornl.gov:intersect-em/particle_finding.git
cd particle_finding
python -m venv env
source env/bin/activate
```

This prepares your python environment for installation in the next steps

#### Conda pre-installation
To install for the first time using [conda](https://docs.conda.io/en/latest/):
```
conda create -n arm python==3.10
git clone git@code.ornl.gov:arm-inititative/multi-particle-cryoem.git
cd multi-particle-cryoem
```

## Installation

After pre-installation, these are the directions to install the `arm_em` package.

#### Default install (CPU-enabled version)

By default, the package works in CPU-only mode:

```
pip install -e .
```

#### GPU-enabled version

To run the GPU-enabled package you need at least one Nvidia GPU installed on your system. 

Add the following optionatl package to your install command:
```
pip install -e .[gpu]
```

#### Development

To add the development dependencies for code checks, use:
```
pip install -e .[dev]
```

## Package Organization
* The **codes** are located in */src/arm_em/*
* The **notebooks** are located in */notebooks/*
