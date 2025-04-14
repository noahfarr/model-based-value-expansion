import os
import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from parameter_space_noise_jax import (
    NoiseState,
    adapt_noise_state,
    perturb_params,
    ddpg_distance,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "model-based-value-expansion"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 1e-3
    """the learning rate of the actor optimizer"""
    critic_learning_rate: float = 1e-3
    """the learning rate of the critic optimizer"""
    model_learning_rate: float = 1e-3
    """the learning rate of the model optimizer"""
    buffer_size: int = int(1e5)  # TODO: This parameter is not mentioned in the paper
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.001  # TODO: This parameter is not mentioned in the paper
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(1e4)
    """timestep to start learning"""
    model_learning_starts: int = int(5e3)
    """timestep to start learning the model"""
    gradient_steps: int = 4
    """the number of gradient steps"""
    prediction_horizon: int = 2
    """the horizon of the prediction"""
    critic_weight_decay: float = 1e-2
    """the weight decay of the critic"""
    noise_adaptation_frequency: int = 50
    """the frequency of noise adaptation"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, exclude_current_positions_from_observation=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env.action_space.seed(seed)
        return env

    return thunk


@jax.jit
def half_cheetah_v4_reward(obs, action, next_obs):
    velocity = (next_obs[0] - obs[0]) / 0.05
    control_cost = 0.1 * jnp.sum(jnp.square(action))
    return velocity - control_cost


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class Model(nn.Module):
    obs_dim: int

    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.obs_dim)(x)
        return x


@jax.jit
def generate_trajectory(actor_state, model_state, obs):

    def step(obs, _):
        actions = actor.apply(actor_state.target_params, obs)
        delta_next_obs = model.apply(model_state.params, obs, actions)
        next_obs = obs + delta_next_obs
        rewards = jax.vmap(half_cheetah_v4_reward)(obs, actions, next_obs)
        transition = Transition(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rewards,
        )
        return next_obs, transition

    _, transitions = jax.lax.scan(step, obs, None, length=args.prediction_horizon)
    return transitions


class TrainState(TrainState):
    target_params: Optional[flax.core.FrozenDict] = None
    noisy_params: Optional[flax.core.FrozenDict] = None
    adaptation_params: Optional[flax.core.FrozenDict] = None


class Transition(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, model_key, noise_key = jax.random.split(key, 5)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )
    noise_state = NoiseState(param_std=0.2, target_action_std=0.2)

    discounts = args.gamma ** jnp.arange(args.prediction_horizon + 1).reshape(
        args.prediction_horizon + 1, 1, 1
    )

    i, j = jnp.tril_indices(args.prediction_horizon + 1)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        noisy_params=perturb_params(noise_key, actor.init(actor_key, obs), noise_state),
        adaptation_params=perturb_params(
            noise_key, actor.init(actor_key, obs), noise_state
        ),
        tx=optax.adam(learning_rate=args.actor_learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adamw(
            learning_rate=args.critic_learning_rate,
            weight_decay=args.critic_weight_decay,
        ),
    )
    model = Model(
        obs_dim=np.prod(envs.single_observation_space.shape),
    )
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(model_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.model_learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)
    model.apply = jax.jit(model.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        trajectory: Transition,
        discounts: jnp.ndarray,
        i: jnp.ndarray,
        j: jnp.ndarray,
    ):

        final_state_actions = actor.apply(
            actor_state.target_params,
            trajectory.next_observations[args.prediction_horizon],
        ).clip(-1, 1)
        qf1_final_target = qf.apply(
            qf1_state.target_params,
            trajectory.next_observations[args.prediction_horizon],
            final_state_actions,
        ).reshape(-1)

        rewards = jnp.zeros(
            (args.prediction_horizon + 1, args.prediction_horizon + 1, args.batch_size)
        )

        rewards = rewards.at[i, j].set(trajectory.rewards[j - i, :])

        discounted_rewards = discounts * rewards
        discounted_returns = jnp.sum(discounted_rewards, axis=0)

        final_q_values = discounted_returns + args.gamma ** (
            args.prediction_horizon + 1
        ) * (qf1_final_target)

        def mse_loss(params):
            qf_a_values = qf.apply(
                params, trajectory.observations, trajectory.actions
            ).squeeze()
            return ((qf_a_values - final_q_values) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_value), grads = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads)

        return qf1_state, qf1_loss_value, qf1_a_value

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(
                qf1_state.params, observations, actor.apply(params, observations)
            ).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params, actor_state.target_params, args.tau
            )
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params, qf1_state.target_params, args.tau
            )
        )
        return actor_state, qf1_state, actor_loss_value

    @jax.jit
    def update_model(
        model_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
    ):
        delta_next_obs = next_observations - observations

        def mse_loss(params):
            pred_delta_next_obs = model.apply(params, observations, actions)
            return ((pred_delta_next_obs - delta_next_obs) ** 2).mean()

        model_loss_value, grads = jax.value_and_grad(mse_loss)(model_state.params)
        model_state = model_state.apply_gradients(grads=grads)
        return model_state, model_loss_value

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = actor.apply(actor_state.noisy_params, obs)
            actions = np.array(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            key, noise_key = jax.random.split(key)
            actor_state = actor_state.replace(
                noisy_params=perturb_params(noise_key, actor_state.params, noise_state)
            )
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % args.noise_adaptation_frequency == 0:
            data = rb.sample(args.batch_size)

            key, noise_key = jax.random.split(key)
            actor_state = actor_state.replace(
                adaptation_params=perturb_params(
                    noise_key, actor_state.params, noise_state
                )
            )
            actions = actor.apply(actor_state.params, data.observations.numpy())
            noisy_actions = actor.apply(
                actor_state.adaptation_params, data.observations.numpy()
            )
            distance = ddpg_distance(actions, noisy_actions).item()
            noise_state = adapt_noise_state(noise_state, distance)

            writer.add_scalar("noise/param_std", noise_state.param_std, global_step)
            writer.add_scalar("noise/distance", distance, global_step)

        # ALGO LOGIC: training.
        for _ in range(args.gradient_steps):
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                transition = Transition(
                    observations=data.observations.numpy(),
                    actions=data.actions.numpy(),
                    next_observations=data.next_observations.numpy(),
                    rewards=data.rewards.flatten().numpy(),
                )
                transition = jax.tree.map(lambda x: x[None, ...], transition)
                transitions = generate_trajectory(
                    actor_state,
                    model_state,
                    data.next_observations.numpy(),
                )
                trajectory = jax.tree.map(
                    lambda transition, trajectory: jnp.concatenate(
                        [transition, trajectory], axis=0
                    ),
                    transition,
                    transitions,
                )

                qf1_state, qf1_loss_value, qf1_a_value = update_critic(
                    actor_state, qf1_state, trajectory, discounts, i, j
                )

                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/qf1_loss", qf1_loss_value.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss_value.item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf1_a_value", qf1_a_value.item(), global_step
                    )

            if global_step > args.model_learning_starts:
                data = rb.sample(args.batch_size)
                model_state, model_loss_value = update_model(
                    model_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/model_loss", model_loss_value.item(), global_step
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DDPG",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
