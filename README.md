# BATPAL: Bayesian Robust Cooperative Multi-Agent Reinforcement Learning Against Unknown Adversaries

Official implementation for the ICLR 2026 paper on Bayesian robust cooperative MARL under unknown adversarial objectives.

The method in this repository is **BATPAL** (Bayesian Type-Partitioned Adversarial Learning), which models deployment-time uncertainty over attacker objectives as a Bayesian game and learns cooperative policies that adapt across adversarial severity types.


## 1. Repository Layout

```
BATPAL/
  algo/                 # policy/algorithm implementations
  runner/               # training loops and experiment runners
  env/                  # environment wrappers and custom environments
  configs/algo/         # algorithm configs
  configs/env/          # environment configs
  pretrained/           # pretrained baseline policies used by BATPAL
main.py                 # experiment entrypoint
requirements.txt        # python dependencies
```

## 2. Installation

### 2.1 Requirements

```bash

pip install -r requirements.txt
```

### 2.2 Environment-specific dependencies

- **SMAC** requires StarCraft II assets and SMAC installation. Please see [SMAC](https://github.com/oxwhirl/smac) for more details. 


## 3. Running Experiments

Entrypoint:

```bash
python main.py --algo <algorithm> --env <environment> --exp_name <name>
```

### 3.1 Example: Train BATPAL on SMAC 2s3z

```bash
python main.py --algo mappo_multi_type_belief --env smac --exp_name batpal_2s3z --seed 1 --map_name 2s3z
```

Notes:
- Default BATPAL config is in `BATPAL/configs/algo/mappo_multi_type_belief.yaml`.
- This config uses pretrained benign policies by default:
  `BATPAL/pretrained/baseline_policies/2s3z/models`.

### 3.2 Train benign baseline (no adversary)

```bash
python main.py --algo mappo_no_adv --env smac --exp_name benign_2s3z --seed 1 --map_name 2s3z
```

### 3.3 Evaluate a trained checkpoint

Example:

```bash
python main.py --algo mappo_multi_type_belief --env smac --exp_name eval_2s3z --map_name 2s3z --eval_only True --model_dir BATPAL/results/smac/2s3z/mappo_multi_type_belief/batpal_2s3z/1/run1/models/500000
```

## 4. Configuration Guide

Main algorithm knobs (`BATPAL/configs/algo/mappo_multi_type_belief.yaml`):
- `n_severity_types`: number of adversarial severity partitions
- `v_min`, `v_max`: reward range used for type partitioning
- `log_barrier_coef`: external-constraint regularization strength
- `baseline_policy`: path to pretrained benign policy used by BATPAL
- `adv_prob`: probability of adversarial episodes during training

Main environment knobs:
- `BATPAL/configs/env/smac.yaml`: SMAC map and state settings
- `BATPAL/configs/env/lbforaging.yaml`: LBF field/players/food
- `BATPAL/configs/env/pettingzoo_mpe.yaml`: MPE scenario and horizon

## 5. Logging and Outputs

Results are saved under:

`BATPAL/results/<env>/<scenario>/<algo>/<exp_name>/<seed>/runX/`

Each run contains:
- `config.json`: full resolved config for reproducibility
- `logs/`: TensorBoard-compatible logs and `summary.json`
- `models/`: model checkpoints


## 6. Citation

If you use this repository, please cite 
Kiarash Kazari, György Dán, ``Bayesian Robust Cooperative Multi-Agent Reinforcement Learning Against Unknown Adversaries'',
in Proc. of ICLR, 2026

