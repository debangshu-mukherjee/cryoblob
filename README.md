# Multi-Particle Cryo-EM

This package is intended to function as the repository for data wrangling and analysis.

This package uses **GPUs** to process the data using the `cupy` package. Thus to install
and run the package you need at least one Nvidia GPU installed on your system. 

Additionally, the package organization assumes that a POSIX compliant system is present *(Linux or MacOS)*. Ideally you should be running this in a Linux environment.

## Installation
To install for the first time:
```
conda create -n arm python==3.10
git clone git@code.ornl.gov:arm-inititative/multi-particle-cryoem.git
cd multi-particle-cryoem
pip install -e .
```

After the first time setup, run as:

```
cd multi-particle-cryoem
pip install -e .
```
## Package Organization
* The **codes** are located in */src/arm_em/*
* The **notebooks** are located in */notebooks/*
* The **data** are located in */data/*
* The **processed data** are located in */data/results/*
