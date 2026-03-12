# Model-Based Value Expansion

A JAX/Flax implementation of Model-Based Value Expansion (MVE) with DDPG for continuous control tasks.

MVE improves DDPG by learning a dynamics model of the environment and using it to generate multi-step rollouts for more accurate value estimation. Rather than relying solely on single-step Bellman backups, the critic is trained with targets that include model-predicted rewards over a configurable horizon, reducing value estimation bias.

Based on the paper: [Value Prediction Network](https://arxiv.org/abs/1803.00101) (Feinberg et al., 2018).

## Architecture

- **Actor**: Policy network with LayerNorm, outputs continuous actions via `tanh` squashing
- **Critic (Q-Network)**: State-action value estimator with LayerNorm and weight decay
- **Dynamics Model**: 8-layer MLP that predicts observation deltas given state-action pairs
- **Exploration**: Parameter space noise with adaptive magnitude (Plappert et al., 2017)

## Installation

Requires Python 3.10+.

```bash
poetry install
```

## Usage

```bash
# Default: HalfCheetah-v4, 1M timesteps
poetry run python mve_ddpg_continuous_action_jax.py

# Custom environment and seed
poetry run python mve_ddpg_continuous_action_jax.py --env-id Walker2d-v4 --seed 42

# With Weights & Biases tracking and video capture
poetry run python mve_ddpg_continuous_action_jax.py --track --capture-video
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--env-id` | `HalfCheetah-v4` | Gymnasium environment |
| `--total-timesteps` | `1000000` | Total training steps |
| `--prediction-horizon` | `2` | Model rollout horizon for value expansion |
| `--batch-size` | `512` | Replay buffer sample size |
| `--gradient-steps` | `4` | Gradient updates per environment step |
| `--gamma` | `0.99` | Discount factor |
| `--tau` | `0.001` | Target network soft update rate |

Run `--help` for the full list of options.

## Logging

Training metrics are logged to TensorBoard under `runs/`. Launch the dashboard with:

```bash
tensorboard --logdir runs/
```

## License

MIT
