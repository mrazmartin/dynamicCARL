# DynamicCARL

**Dynamic Context Scheduling for Contextual Reinforcement Learning**

Code accompanying the paper:
> *Dynamic Context Scheduling: Learning Beyond the Static Universe*
> Martin Mráz, André Biedenkapp — University of Freiburg

---

## What is this?

Standard contextual RL assigns each episode a fixed physics parameter (pole length, gravity, payload mass, …) sampled from a distribution. **DynamicCARL** instead lets that parameter *evolve within the episode* according to a schedule — sinusoidal sweeps, random walks, piecewise jumps, cosine annealing, and more.

This exposes agents to a much richer distribution of transitions within a single episode and is a simple drop-in on top of any [CARL](https://github.com/automl/CARL) environment.

## Installation

```bash
# Clone this repo
git clone https://github.com/mrazmartin/dynamicCARL.git
cd dynamicCARL

# Install the package
pip install -e .

# CARL (not on PyPI)
pip install git+https://github.com/automl/CARL.git

# Box2D environments
pip install swig box2d-py

# Pretrained model demos
pip install stable-baselines3

# Interactive CarRacing window
pip install pygame

# MuJoCo environments (Go2 / Quadruped) — use --no-deps to avoid labmaze/Bazel
pip install mujoco
pip install dm-control --no-deps
pip install dm-env pyopengl
```

## Quick start

See [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) for a full walkthrough:

1. CartPole with sinusoidal pole-length schedule (pretrained PPO)
2. Comparing all schedule families side-by-side
3. BipedalWalker with dynamic payload offset (pretrained PPO)
4. CarRacing with dynamic center-of-mass shift
5. Animated rollout viewer
6. Interactive play windows (local only)
7. Custom scheduler example
8. MuJoCo Go2 with step-function gravity switch

## Core API

```python
from dynamic_carl.env_wrappers import GymDynamicContextCarlWrapper
from dynamic_carl.gym_context_updates import make_sinusoidal

scheduler = make_sinusoidal("length", amplitude=0.2, period=100,
                             min_val=0.1, max_val=1.2, seed=0)

env = GymDynamicContextCarlWrapper(
    base_carl_env,
    feature_update_fns={"length": scheduler},
    ctx_getters={"length": my_getter},
    ctx_setters={"length": my_setter},
)
```

Available schedules: `make_sinusoidal`, `make_cosine_annealing`, `make_random_walk`,
`make_piecewise_constant`, `make_sudden_jump`, `make_levy_walk`, `make_identity`.

## Pretrained models

Small checkpoints for the notebook demos are in `models/`:
- `cartpole_ppo_no_ctx_150k.zip` — CartPole, 150k steps, no context in obs
- `walker_ppo_com_x_3M.zip` — BipedalWalker, 3M steps, cosine-annealing COM_X schedule

## Experiments

Training scripts and configs to reproduce the paper's results are in `experiments/`.

## Citation

```bibtex
@inproceedings{mraz2025dynamic,
  title={Dynamic Context Scheduling: Learning Beyond the Static Universe},
  author={Mráz, Martin and Biedenkapp, André},
  booktitle={EWRL 2025},
  year={2025}
}
```
