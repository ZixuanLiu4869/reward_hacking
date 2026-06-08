# Robust Optimization for Mitigating Reward Hacking with Correlated Proxies

This repository contains the official code for the paper:

**Robust Optimization for Mitigating Reward Hacking with Correlated Proxies**
Zixuan Liu, Xiaolin Sun, and Zizhan Zheng
ICLR 2026

OpenReview: https://openreview.net/forum?id=O3shkBWM2s

## Overview

Reward hacking occurs when an agent exploits misspecified proxy rewards instead of optimizing the intended true objective. This repository implements robust optimization methods for mitigating reward hacking when the proxy reward is correlated with, but not identical to, the true reward.

The code builds on the ORPO codebase:

https://github.com/cassidylaidlaw/orpo

Please refer to the original ORPO repository for additional details on environment setup and implementation structure. In this repository, the main Python code is located under the `occupancy_measures` package.

## Installation

Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Setup

This repository supports the following environments:

* `traffic`
* `pandemic`
* `glucose`
* `tomato level=4`

### Traffic Environment

The traffic environment depends on SUMO. On Ubuntu, SUMO can usually be installed with:

```bash
sudo apt install sumo sumo-tools sumo-doc
```

For more details, please refer to the original traffic environment repository:

https://github.com/shivamsinghal001/flow_reward_misspecification

### Pandemic Environment

The pandemic environment is available here:

https://github.com/shivamsinghal001/pandemic

### Glucose Environment

The glucose environment is available here:

https://github.com/shivamsinghal001/glucose

### Tomato Environment

For the tomato environment, please rename:

```text
occupancy_measures/agents/orpo_tomato.py
```

to:

```text
occupancy_measures/agents/orpo.py
```

This ensures that tomato training uses the correct implementation. For all other environments, training uses the original `orpo.py` file.

## Algorithms

The algorithm switches are defined in:

```text
occupancy_measures/agents/orpo.py
```

Inside the `training_step` function, around line 1458, you will find:

```python
def training_step(self) -> ResultDict:
    ORPO_TRAINING = True
    ORPO_DISC_TRAINING = False
    LINEAR_MAX_MIN = False
    MAX_MIN = False
```

Set the corresponding variable to `True` to run a specific algorithm.

* `ORPO_TRAINING=True`: runs the original ORPO algorithm.
* `ORPO_DISC_TRAINING=True`: runs ORPO*, the variant used in our paper.
* `LINEAR_MAX_MIN=True`: runs the linear max-min robust optimization variant.
* `MAX_MIN=True`: runs the max-min robust optimization variant.

Please make sure that only the intended algorithm flag is set to `True` for each run.

## Running Experiments

To run training, use:

```bash
python -m occupancy_measures.experiments.orpo_experiments \
  with env_to_run=$ENV \
  reward_fun=proxy \
  exp_algo=ORPO \
  'om_divergence_coeffs=['$COEFF']' \
  'checkpoint_to_load_policies=["'$BC_CHECKPOINT'"]' \
  checkpoint_to_load_current_policy=$BC_CHECKPOINT \
  seed=$SEED \
  experiment_tag=state-action \
  'om_divergence_type=["'$TYPE'"]'
```

where:

* `$ENV` is the environment name.
* `$COEFF` is the divergence coefficient.
* `$BC_CHECKPOINT` is the checkpoint path.
* `$SEED` is the random seed.
* `$TYPE` is the occupancy-measure divergence type.

Supported values of `$ENV` include:

```text
traffic
pandemic
glucose
tomato level=4
```

Supported value of `$TYPE`:

```text
sqrt_chi2
```

## Example Command

An example command is:

```bash
python -m occupancy_measures.experiments.orpo_experiments \
  with env_to_run=traffic \
  reward_fun=proxy \
  exp_algo=ORPO \
  'om_divergence_coeffs=[0.1]' \
  'checkpoint_to_load_policies=["path/to/bc_checkpoint"]' \
  checkpoint_to_load_current_policy=path/to/bc_checkpoint \
  seed=0 \
  experiment_tag=state-action \
  'om_divergence_type=["sqrt_chi2"]'
```

Please replace `path/to/bc_checkpoint` with the actual checkpoint path.


## Notes

This repository adapts code from the original ORPO implementation:

https://github.com/cassidylaidlaw/orpo

Please also consult the original repository for additional installation details, environment dependencies, and implementation background.

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
liu2026robust,
title={Robust Optimization for Mitigating Reward Hacking with Correlated Proxies},
author={Zixuan Liu and Xiaolin Sun and Zizhan Zheng},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=O3shkBWM2s}
}
```

## Acknowledgements

This codebase is adapted from the ORPO repository:

https://github.com/cassidylaidlaw/orpo

We thank the authors of ORPO and the environment repositories for making their code publicly available.
