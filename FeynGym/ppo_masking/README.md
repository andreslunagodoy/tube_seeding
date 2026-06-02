# ppo_masking

`ppo_masking` is a small internal Python package containing a fork of
[CleanRL](https://github.com/vwxyzjn/cleanrl)'s single-file PPO implementation,
adapted for discrete action masking.
It is used by the FeynGym examples to train policies when only a subset of
actions is legal at each environment state.

The main public entry points are:

- `ppo_masking.Args`: dataclass of PPO hyperparameters and runtime options;
- `ppo_masking.run(args)`: train a PPO agent and return the trained `Agent`;
- `ppo_masking.test_total_reward(...)`: greedily evaluate a trained actor;
- `ppo_masking.layer_init(...)`: helper for orthogonal layer initialization.

## Installation

From the repository root:

```bash
cd ppo_masking
pip install -e .
```

This installs the package and dependencies, `gymnasium`, `torch`, `numpy`,
`tyro`, and `tensorboard`.

## Action-Mask Contract

This package is intended for `gymnasium` environments with `Discrete` action
spaces. The environment should return an action mask in the `info` dictionary
from both `reset()` and `step()`:

```python
obs, info = env.reset()
info["action_mask"]  # shape: (n_actions,), dtype bool-like

obs, reward, terminated, truncated, info = env.step(action)
info["action_mask"]  # shape: (n_actions,), dtype bool-like
```

For vectorized training, Gymnasium stacks these masks into shape
`(num_envs, n_actions)`. `True` means the action is valid, and `False` means it
is masked out.

The training loop can fall back to an all-true mask if the environment omits the
mask during rollout, but evaluation requires `info["action_mask"]` and asserts
that it is present. In normal use with this package, environments should always
provide the mask.

## Network Requirements

`ppo_masking` does not construct actor and critic networks automatically. Pass
them through `Args(actor=..., critic=...)`.

For an environment with observation shape `obs_shape` and `Discrete(n_actions)`,
the networks should satisfy:

- `actor(obs_batch)` returns logits with shape `(batch, n_actions)`;
- `critic(obs_batch)` returns values with shape `(batch, 1)` or another shape
  flattenable to `(batch,)`.

The actor receives raw observations converted to floating-point tensors. If the
environment observation is image-like, the actor should handle that layout
itself.

## Minimal Example

The repository includes a tiny masked environment in
`examples/three_position_env.py`. Importing the file registers
`ThreePositionEnv-v0`.

```python
import sys

sys.path.insert(0, "examples")

import three_position_env
import ppo_masking
import torch.nn as nn

actor = nn.Sequential(
    ppo_masking.layer_init(nn.Linear(1, 64)),
    nn.Tanh(),
    ppo_masking.layer_init(nn.Linear(64, 64)),
    nn.Tanh(),
    ppo_masking.layer_init(nn.Linear(64, 2), std=0.01),
)

critic = nn.Sequential(
    ppo_masking.layer_init(nn.Linear(1, 64)),
    nn.Tanh(),
    ppo_masking.layer_init(nn.Linear(64, 64)),
    nn.Tanh(),
    ppo_masking.layer_init(nn.Linear(64, 1), std=1.0),
)

args = ppo_masking.Args(
    env_id="ThreePositionEnv-v0",
    gamma=0.9,
    actor=actor,
    critic=critic,
    total_timesteps=2000,
)

trained_agent = ppo_masking.run(args)
ppo_masking.test_total_reward(
    "ThreePositionEnv-v0",
    trained_agent.actor,
    max_eval_steps=20,
)
```

See `examples/ppo_masking_example.ipynb` for the notebook version.

## Restrictions And Caveats

- Only `gymnasium.spaces.Discrete` action spaces are supported.
- Environments used for evaluation must provide `info["action_mask"]`.
- The mask must have one entry per discrete action.
- The package is designed as an internal helper for this repository, not as a
  general-purpose RL library.
- There is no console script or `python -m ppo_masking` entry point; use it as an
  importable package.
- Training writes TensorBoard logs under `runs/`.
