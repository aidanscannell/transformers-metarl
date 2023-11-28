#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
from dataclasses import dataclass

import click
import d4rl  # need to get envs into gym.make()
import gym
import hydra
import numpy as np
from prettytable import PrettyTable
import torch
from typing import Optional

from garage import EnvSpec, wrap_experiment
from garage.envs import GymEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv

# from garage.envs.metaworld import ML1Env
from garage.experiment import OnlineMetaEvaluator  # MetaEvaluator,
from garage.experiment import Snapshotter, task_sampler
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.policies import (  # GaussianMLPPolicy,
    GaussianMemoryTransformerPolicy,
    GaussianTransformerEncoderPolicy,
    GaussianTransformerPolicy,
)
from garage.torch import set_gpu_mode
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

# class MassDampingENV(gym.Env):
# def __init__(self, env):
#     self._env = env
#     self.action_space = env.action_space
#     self.observation_space = env.observation_space
#     self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
#     self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
#     self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
#     self.original_damping = env.env.wrapped_env.model.dof_damping.copy()

# # ind is from 0 to 24
# def reset(self, ind=None):
#     if ind is None:
#         num_tasks = 24
#         task_idxs = np.arange(num_tasks)
#         ind = np.random.choice(task_idxs, 2)
#         print(f"ind {ind}")
#     model = self._env.env.wrapped_env.model
#     n_link = model.body_mass.shape[0]
#     ind_mass = ind // 5
#     ind_damp = ind % 5
#     for i in range(n_link):
#         model.body_mass[i] = self.original_body_mass[i] * self.mass_ratios[ind_mass]
#         model.dof_damping[i] = (
#             self.original_damping[i] * self.damping_ratios[ind_damp]
#         )
#     return self._env.reset()

# def step(self, action):
#     return self._env.step(action)

# def get_normalized_score(self, score):
#     return self._env.get_normalized_score(score)

# def sample_tasks(self, num_tasks):
#     """Sample a list of `num_tasks` tasks.

#     Args:
#         num_tasks (int): Number of tasks to sample.

#     Returns:
#         list[dict[str, float]]: A list of "tasks," where each task is a
#             dictionary containing a single key, "direction", mapping to -1
#             or 1.

#     """
#     num_tasks = 24
#     task_idxs = np.arange(num_tasks)
#     tasks = np.random.choice(task_idxs, 2)
#     return tasks

# def set_task(self, task):
#     """Reset with a task.

#     Args:
#         task (dict[str, float]): A task (a dictionary containing a single
#             key, "direction", mapping to -1 or 1).

#     """
#     self.reset(ind=task)


class MassDampingENV(gym.Env):
    # class HopperMediumV2(gym.Env):
    def __init__(self, env, task_idx: int = 0):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
        self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
        self.original_damping = env.env.wrapped_env.model.dof_damping.copy()

        self.num_tasks = 25
        self.task_idxs = np.arange(self.num_tasks)
        self.task_idx = task_idx
        self._reset(ind=self.task_idx)

        # self.spec = EnvSpec(
        #     action_space=self.action_space, observation_space=self.observation_space
        # )

    # ind is from 0 to 24
    def reset(self):
        ind = np.random.choice(self.task_idxs, 1)
        # print(f"ind {ind}")
        return self._reset(ind=ind)

    def _reset(self, ind):
        if isinstance(ind, np.ndarray):
            ind = ind.item()
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
        tasks = np.random.choice(self.task_idxs, num_tasks)
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "direction", mapping to -1 or 1).

        """
        self._reset(ind=task)


class HopperMediumV2(MassDampingENV):
    def __init__(self, task=0):
        env = gym.make("hopper-medium-v2")
        super().__init__(env=env, task_idx=task)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_env(env_name):
    try:
        m = __import__("garage")
        m = getattr(m, "envs")
        m = getattr(m, "mujoco")
        return getattr(m, env_name)
    except:
        m = __import__("garage")
        m = getattr(m, "envs")
        # m = getattr(m, "metaworld")
        return getattr(m, env_name)


# @click.command()
# @click.option("--env_name", default="ML1ReachEnv")
# @click.option("--seed", default=1)
# @click.option("--max_episode_length", default=200)
# @click.option("--meta_batch_size", default=20)
# @click.option("--n_epochs", default=5000)
# @click.option("--episode_per_task", default=2)
# @click.option("--wm_embedding_hidden_size", default=32)
# @click.option("--n_heads", default=1)
# @click.option("--d_model", default=4)
# @click.option("--layers", default=2)
# @click.option("--dropout", default=0.0)
# @click.option("--wm_size", default=2)
# @click.option("--em_size", default=1)
# @click.option("--dim_ff", default=16)
# @click.option("--discount", default=0.99)
# @click.option("--gae_lambda", default=0.95)
# @click.option("--lr_clip_range", default=0.2)
# @click.option("--policy_lr", default=2.5e-4)
# @click.option("--vf_lr", default=2.5e-4)
# @click.option("--minibatch_size", default=32)
# @click.option("--max_opt_epochs", default=10)
# @click.option("--center_adv", is_flag=True)
# @click.option("--positive_adv", is_flag=True)
# @click.option("--policy_ent_coeff", default=0.02)
# @click.option("--use_softplus_entropy", is_flag=True)
# @click.option("--stop_entropy_gradient", is_flag=True)
# @click.option("--entropy_method", default="regularized")
# @click.option("--share_network", is_flag=True)
# @click.option("--architecture", default="Encoder")
# @click.option("--policy_head_input", default="latest_memory")
# @click.option("--dropatt", default=0.0)
# @click.option("--attn_type", default=1)
# @click.option("--pre_lnorm", is_flag=True)
# @click.option("--init_params", is_flag=True)
# @click.option("--gating", default="residual")
# @click.option("--init_std", default=1.0)
# @click.option("--learn_std", is_flag=True)
# @click.option("--policy_head_type", default="Default")
# @click.option("--policy_lr_schedule", default="no_schedule")
# @click.option("--vf_lr_schedule", default="no_schedule")
# @click.option("--decay_epoch_init", default=500)
# @click.option("--decay_epoch_end", default=1000)
# @click.option("--min_lr_factor", default=0.1)
# @click.option("--tfixup", is_flag=True)
# @click.option("--remove_ln", is_flag=True)
# @click.option("--recurrent_policy", is_flag=True)
# @click.option("--pretrained_dir", default=None)
# @click.option("--pretrained_epoch", default=4980)
# @click.option("--output_weights_scale", default=1.0)
# @click.option("--normalized_wm", is_flag=True)
# @click.option("--annealing_std", is_flag=True)
# @click.option("--min_std", default=1e-6)
# @click.option("--gpu_id", default=0)
@wrap_experiment(snapshot_mode="gap", snapshot_gap=30)
def transformer_ppo_halfcheetah(
    ctxt,
    env_name,
    seed,
    max_episode_length,
    meta_batch_size,
    n_epochs,
    episode_per_task,
    wm_embedding_hidden_size,
    n_heads,
    d_model,
    layers,
    dropout,
    wm_size,
    em_size,
    dim_ff,
    discount,
    gae_lambda,
    lr_clip_range,
    policy_lr,
    vf_lr,
    minibatch_size,
    max_opt_epochs,
    center_adv,
    positive_adv,
    policy_ent_coeff,
    use_softplus_entropy,
    stop_entropy_gradient,
    entropy_method,
    share_network,
    architecture,
    policy_head_input,
    dropatt,
    attn_type,
    pre_lnorm,
    init_params,
    gating,
    init_std,
    learn_std,
    policy_head_type,
    policy_lr_schedule,
    vf_lr_schedule,
    decay_epoch_init,
    decay_epoch_end,
    min_lr_factor,
    recurrent_policy,
    tfixup,
    remove_ln,
    pretrained_dir,
    pretrained_epoch,
    output_weights_scale,
    normalized_wm,
    annealing_std,
    min_std,
    gpu_id,
):
    """Train PPO with HalfCheetah environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    print(f"wm_embedding_hidden_size {wm_embedding_hidden_size}")
    breakpoint()
    set_seed(seed)

    policy = None
    value_function = None
    if pretrained_dir is not None:
        snapshotter = Snapshotter()
        data = snapshotter.load(pretrained_dir, itr=pretrained_epoch)
        policy = data["algo"].policy
        value_function = data["algo"].value_function

    trainer = Trainer(ctxt)
    # env_class = get_env(env_name)
    # env_class = MassDampingENV
    env_class = HalfCheetahVelEnv
    env_class = HopperMediumV2

    env_name = "hopper-medium-v2"
    # env = gym.make(env_name)

    # def make_env():
    #     env = gym.make(env_name)
    #     env = MassDampingENV(env)
    #     # env.reset(0)  # between 0 and 24
    #     return env

    def env_wrapper(env, *args):
        env = RL2Env(GymEnv(env, max_episode_length=max_episode_length))
        # env = MassDampingENV(env)
        # wrapper = lambda env, _: normalize(
        #     GymEnv(env, max_episode_length=max_episode_length)
        # )
        # task = 0
        # env = wrapper(env, task)
        # env.reset(task)  # between 0 and 24
        return env

    tasks = task_sampler.SetTaskSampler(
        env_class,
        # make_env,
        # wrapper=env_wrapper,
        wrapper=lambda env, _: RL2Env(
            GymEnv(env, max_episode_length=max_episode_length)
        ),
    )

    env_spec = RL2Env(GymEnv(env_class(), max_episode_length=max_episode_length)).spec
    print(f"env_spec {env_spec}")

    if annealing_std:
        annealing_rate = (min_std / init_std) ** (
            3.0 / (meta_batch_size * 2 * n_epochs)
        )  # reach min step at 2/3 * n_epochs
    else:
        annealing_rate = 1.0

    if architecture == "Encoder":
        policy = (
            GaussianTransformerEncoderPolicy(
                name="policy",
                env_spec=env_spec,
                encoding_hidden_sizes=(wm_embedding_hidden_size,),
                nhead=n_heads,
                d_model=d_model,
                num_encoder_layers=layers,
                dropout=dropout,
                obs_horizon=wm_size,
                dim_feedforward=dim_ff,
                policy_head_input=policy_head_input,
                policy_head_type=policy_head_type,
                tfixup=tfixup,
                remove_ln=remove_ln,
                init_std=init_std,
                learn_std=learn_std,
                min_std=min_std,
                annealing_rate=annealing_rate,
                mlp_output_w_init=lambda x: torch.nn.init.xavier_uniform_(
                    x, gain=output_weights_scale
                ),
                normalize_wm=normalized_wm,
                recurrent_policy=recurrent_policy,
            )
            if policy is None
            else policy
        )
    elif architecture == "Transformer":
        policy = GaussianTransformerPolicy(
            name="policy",
            env_spec=env_spec,
            encoding_hidden_sizes=(wm_embedding_hidden_size,),
            nhead=n_heads,
            d_model=d_model,
            num_decoder_layers=layers,
            num_encoder_layers=layers,
            dropout=dropout,
            obs_horizon=wm_size,
            hidden_horizon=em_size,
            dim_feedforward=dim_ff,
        )
    elif architecture == "MemoryTransformer":
        policy = GaussianMemoryTransformerPolicy(
            name="policy",
            env_spec=env_spec,
            encoding_hidden_sizes=(wm_embedding_hidden_size,),
            nhead=n_heads,
            d_model=d_model,
            num_encoder_layers=layers,
            dropout=dropout,
            dropatt=dropatt,
            obs_horizon=wm_size,
            mem_len=em_size,
            dim_feedforward=dim_ff,
            attn_type=attn_type,
            pre_lnorm=pre_lnorm,
            init_params=init_params,
            gating=gating,
            init_std=init_std,
            learn_std=learn_std,
            policy_head_type=policy_head_type,
            policy_head_input=policy_head_input,
        )

    # count_parameters(policy)

    base_model = policy if share_network else None

    value_function = GaussianMLPValueFunction(
        env_spec=env_spec,
        base_model=base_model,
        hidden_sizes=(64, 64),
        learn_std=learn_std,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    # count_parameters(value_function)

    meta_evaluator = OnlineMetaEvaluator(
        test_task_sampler=tasks,
        n_test_tasks=30,
        n_test_episodes=1,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=2),
    )
    # meta_evaluator = None

    steps_per_epoch = (
        max_opt_epochs
        * (max_episode_length * episode_per_task * meta_batch_size)
        // minibatch_size
    )

    algo = RL2PPO(
        meta_batch_size=meta_batch_size,
        task_sampler=tasks,
        env_spec=env_spec,
        policy=policy,
        value_function=value_function,
        episodes_per_trial=episode_per_task,
        discount=discount,
        gae_lambda=gae_lambda,
        lr_clip_range=lr_clip_range,
        policy_lr=policy_lr,
        vf_lr=vf_lr,
        minibatch_size=minibatch_size,
        max_opt_epochs=max_opt_epochs,
        use_softplus_entropy=use_softplus_entropy,
        stop_entropy_gradient=stop_entropy_gradient,
        entropy_method=entropy_method,
        policy_ent_coeff=policy_ent_coeff,
        center_adv=center_adv,
        positive_adv=positive_adv,
        meta_evaluator=meta_evaluator,
        policy_lr_schedule=policy_lr_schedule,
        vf_lr_schedule=vf_lr_schedule,
        decay_epoch_init=decay_epoch_init,
        decay_epoch_end=decay_epoch_end,
        min_lr_factor=min_lr_factor,
        steps_per_epoch=steps_per_epoch,
        n_epochs=n_epochs,
        n_epochs_per_eval=5,
    )

    if torch.cuda.is_available() and gpu_id >= 0:
        set_gpu_mode(True, gpu_id)
    else:
        set_gpu_mode(False)
    algo.to()

    trainer.setup(
        algo,
        tasks.sample(meta_batch_size),
        sampler_cls=LocalSampler,
        n_workers=meta_batch_size,
        worker_class=RL2Worker,
        worker_args=dict(n_episodes_per_trial=episode_per_task),
    )

    trainer.train(
        n_epochs=n_epochs,
        batch_size=episode_per_task * max_episode_length * meta_batch_size,
    )


@dataclass
class TrainConfig:
    # defaults: List[Any] = field(
    #     default_factory=lambda: [
    #         {"agent": "ddpg"},
    #         {"env": "cartpole_swingup"},
    #         # {
    #         #     "override hydra/launcher": "triton_config",  # Use slurm (on cluster) for multirun
    #         # "override hydra/launcher": "slurm",  # Use slurm (on cluster) for multirun
    #         # "override hydra/launcher": "triton_config",  # Use slurm (on cluster) for multirun
    #         # },
    #     ]
    # )

    _target_: str = "__main__.transformer_ppo_halfcheetah"
    env_name: str = "HalfCheetah"
    seed: int = 1
    max_episode_length: int = 200
    meta_batch_size: int = 20
    n_epochs: int = 5000
    episode_per_task: int = 2
    wm_embedding_hidden_size: int = 32
    n_heads: int = 1
    d_model: int = 4
    layers: int = 2
    dropout: float = 0.0
    wm_size: int = 2
    em_size: int = 1
    dim_ff: int = 16
    discount: float = 0.99
    gae_lambda: float = 0.95
    lr_clip_range: float = 0.2
    policy_lr: float = 2.5e-4
    vf_lr: float = 2.5e-4
    minibatch_size: int = 32
    max_opt_epochs: int = 10
    center_adv: bool = True
    positive_adv: bool = True
    policy_ent_coeff: float = 0.02
    use_softplus_entropy: bool = True
    stop_entropy_gradient: bool = True
    entropy_method: str = "regularized"
    share_network: bool = True
    architecture: str = "Encoder"
    policy_head_input: str = "latest_memory"
    dropatt: float = 0.0
    attn_type: int = 1
    pre_lnorm: bool = True
    init_params: bool = True
    gating: str = "residual"
    init_std: float = 1.0
    learn_std: bool = True
    policy_head_type: str = "Default"
    policy_lr_schedule: str = "no_schedule"
    vf_lr_schedule: str = "no_schedule"
    decay_epoch_init: int = 500
    decay_epoch_end: int = 1000
    min_lr_factor: float = 0.1
    tfixup: bool = True
    remove_ln: bool = True
    recurrent_policy: bool = True
    pretrained_dir: Optional[str] = None
    pretrained_epoch: int = 4980
    output_weights_scale: float = 1.0
    normalized_wm: bool = True
    annealing_std: bool = True
    min_std: float = 1e-6
    gpu_id: int = 0


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="half_cheetah")
def hydra_wrapper(cfg: TrainConfig):
    print(f"cfg {cfg}")
    return hydra.utils.instantiate(cfg)
    # return transformer_ppo_halfcheetah(
    #     env_name=cfg.env_name,
    #     seed=1,
    #     max_episode_length=200,
    #     meta_batch_size=20,
    #     n_epochs=5000,
    #     episode_per_task=2,
    #     wm_embedding_hidden_size=32,
    #     n_heads=1,
    #     d_model=4,
    #     layers=2,
    #     dropout=0.0,
    #     wm_size=2,
    #     em_size=1,
    #     dim_ff=16,
    #     discount=0.99,
    #     gae_lambda=0.95,
    #     lr_clip_range=0.2,
    #     policy_lr=2.5e-4,
    #     vf_lr=2.5e-4,
    #     minibatch_size=32,
    #     max_opt_epochs=10,
    #     center_adv=True,
    #     positive_adv=True,
    #     policy_ent_coeff=0.02,
    #     use_softplus_entropy=True,
    #     stop_entropy_gradient=True,
    #     entropy_method="regularized",
    #     share_network=True,
    #     architecture="Encoder",
    #     policy_head_input="latest_memory",
    #     dropatt=0.0,
    #     attn_type=1,
    #     pre_lnorm=True,
    #     init_params=True,
    #     gating="residual",
    #     init_std=1.0,
    #     learn_std=True,
    #     policy_head_type="Default",
    #     policy_lr_schedule="no_schedule",
    #     vf_lr_schedule="no_schedule",
    #     decay_epoch_init=500,
    #     decay_epoch_end=1000,
    #     min_lr_factor=0.1,
    #     tfixup=True,
    #     remove_ln=True,
    #     recurrent_policy=True,
    #     pretrained_dir=None,
    #     pretrained_epoch=4980,
    #     output_weights_scale=1.0,
    #     normalized_wm=True,
    #     annealing_std=True,
    #     min_std=1e-6,
    #     gpu_id=0,
    # )


if __name__ == "__main__":
    hydra_wrapper()
