#!/usr/bin/env python3
"""PEARL HalfCheetahVel example."""
import click
import gym
import d4rl

from garage.envs import GymEnv, normalize
from garage.envs.mujoco import HalfCheetahVelEnv
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy, TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch import set_gpu_mode
from garage.trainer import Trainer
from garage import wrap_experiment
import numpy as np


class MassDampingENV(gym.Env):
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
        self.original_damping = env.env.wrapped_env.model.dof_damping.copy()

    # ind is from 0 to 24
    def reset(self, ind):
        model = self._env.env.wrapped_env.model
        n_link = model.body_mass.shape[0]
        ind_mass = ind // 5
        ind_damp = ind % 5
        for i in range(n_link):
            model.body_mass[i] = self.original_body_mass[i] * self.mass_ratios[ind_mass]
            model.dof_damping[i] = (
                self.original_damping[i] * self.damping_ratios[ind_damp]
            )
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def get_normalized_score(self, score):
        return self._env.get_normalized_score(score)

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "direction", mapping to -1
                or 1.

        """
        num_tasks = 24
        task_idxs = np.arange(num_tasks)
        tasks = np.random.choice(task_idxs, 2)
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "direction", mapping to -1 or 1).

        """
        self.reset(ind=task)


@click.command()
@click.option("--num_epochs", default=500)
@click.option("--num_train_tasks", default=100)
@click.option("--num_test_tasks", default=100)
@click.option("--encoder_hidden_size", default=200)
@click.option("--net_size", default=300)
@click.option("--num_steps_per_epoch", default=2000)
@click.option("--num_initial_steps", default=2000)
@click.option("--num_steps_prior", default=400)
@click.option("--num_extra_rl_steps_posterior", default=600)
@click.option("--batch_size", default=256)
@click.option("--embedding_batch_size", default=100)
@click.option("--embedding_mini_batch_size", default=100)
@click.option("--max_episode_length", default=200)
@wrap_experiment
def pearl_half_cheetah_vel(
    ctxt=None,
    seed=1,
    num_epochs=500,
    num_train_tasks=100,
    num_test_tasks=100,
    latent_size=5,
    encoder_hidden_size=200,
    net_size=300,
    meta_batch_size=16,
    num_steps_per_epoch=2000,
    num_initial_steps=2000,
    num_tasks_sample=5,
    num_steps_prior=400,
    num_extra_rl_steps_posterior=600,
    batch_size=256,
    embedding_batch_size=100,
    embedding_mini_batch_size=100,
    max_episode_length=200,
    reward_scale=5.0,
    use_gpu=True,
    # use_gpu=False,
):
    """Train PEARL with HalfCheetahVel environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks to use for testing.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_episode_length (int): Maximum episode length.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    set_seed(seed)
    encoder_hidden_sizes = (
        encoder_hidden_size,
        encoder_hidden_size,
        encoder_hidden_size,
    )

    env_name = "hopper-medium-v2"

    def env_wrapper(env, *args):
        env = MassDampingENV(env)
        # wrapper = lambda env, _: normalize(
        #     GymEnv(env, max_episode_length=max_episode_length)
        # )
        task = 0
        # env = wrapper(env, task)
        env.reset(task)  # between 0 and 24
        return env

    # def make_env():
    #     env = gym.make(env_name)
    #     env = MassDampingENV(env)
    #     env.reset(0)  # between 0 and 24
    #     return env

    # create multi-task environment and sample tasks
    env_sampler = SetTaskSampler(
        # make_env,
        HalfCheetahVelEnv,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)
        ),
    )
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(
        HalfCheetahVelEnv,
        # make_env,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)
        ),
    )

    trainer = Trainer(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size]
    )

    vf_env = PEARL.get_env_spec(env[0](), latent_size, "vf")
    vf = ContinuousMLPQFunction(
        env_spec=vf_env, hidden_sizes=[net_size, net_size, net_size]
    )

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size]
    )

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    trainer.setup(
        algo=pearl,
        env=env[0](),
        sampler_cls=LocalSampler,
        sampler_args=dict(max_episode_length=max_episode_length),
        n_workers=1,
        worker_class=PEARLWorker,
    )

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


pearl_half_cheetah_vel()
