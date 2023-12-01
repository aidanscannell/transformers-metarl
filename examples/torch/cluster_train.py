#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
from dataclasses import dataclass, field
from typing import List, Optional, Any

import hydra

from garage import wrap_experiment


@dataclass
class MainConfig:
    # _target_: str = "__main__.main"
    _target_: str = "cluster_train.main"
    env_name: str = "HalfCheetah"  # HopperV2/HalfCheetahV2/Walker2DV2/HalfCheetahVelEnv
    seed: int = 1
    max_episode_length: int = 1000
    # max_episode_length: int = 200
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
    center_adv: bool = False
    positive_adv: bool = False
    policy_ent_coeff: float = 0.02
    use_softplus_entropy: bool = False
    stop_entropy_gradient: bool = False
    entropy_method: str = "regularized"
    share_network: bool = False
    architecture: str = "Encoder"
    policy_head_input: str = "latest_memory"
    dropatt: float = 0.0
    attn_type: int = 1
    pre_lnorm: bool = False
    init_params: bool = False
    gating: str = "residual"
    init_std: float = 1.0
    learn_std: bool = False
    policy_head_type: str = "Default"
    policy_lr_schedule: str = "no_schedule"
    vf_lr_schedule: str = "no_schedule"
    decay_epoch_init: int = 500
    decay_epoch_end: int = 1000
    min_lr_factor: float = 0.1
    tfixup: bool = False
    remove_ln: bool = False
    recurrent_policy: bool = False
    pretrained_dir: Optional[str] = None
    pretrained_epoch: int = 4980
    output_weights_scale: float = 1.0
    normalized_wm: bool = False
    annealing_std: bool = False
    min_std: float = 1e-6
    gpu_id: int = 0


@dataclass
class TrainConfig:
    wandb_run_name: str
    main_config: MainConfig

    defaults: List[Any] = field(
        default_factory=lambda: [{"main_config": "half_cheetah_config"}]
    )

    _target_: str = "__main__.main"
    use_wandb: bool = True
    wandb_project_name: str = "adaptive-context-rl"
    wandb_group: str = "TrMRL"
    wandb_tags: List[str] = field(default_factory=lambda: ["TrMRL"])


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(group="main_config", name="half_cheetah_config", node=MainConfig)
cs.store(name="train_config", node=TrainConfig)


@wrap_experiment(snapshot_mode="gap", snapshot_gap=30)
def main(
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
    import d4rl  # need to get envs into gym.make()
    import gym
    import numpy as np
    from prettytable import PrettyTable
    import torch

    from garage.envs import GymEnv
    from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
    from garage.experiment import OnlineMetaEvaluator, AdaptiveMDPEvaluator
    from garage.experiment import Snapshotter, task_sampler
    from garage.experiment.deterministic import set_seed
    from garage.sampler import LocalSampler
    from garage.torch.algos import RL2PPO
    from garage.torch.algos.rl2 import RL2Env, RL2Worker
    from garage.torch.policies import (
        GaussianMemoryTransformerPolicy,
        GaussianTransformerEncoderPolicy,
        GaussianTransformerPolicy,
    )
    from garage.torch import set_gpu_mode
    from garage.torch.value_functions import GaussianMLPValueFunction
    from garage.trainer import Trainer

    class MassDampingENV(gym.Env):
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
                model.body_mass[i] = (
                    self.original_body_mass[i] * self.mass_ratios[ind_mass]
                )
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

    class HopperV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("hopper-medium-v2")
            super().__init__(env=env, task_idx=task)

    class HalfCheetahV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("halfcheetah-medium-v2")
            super().__init__(env=env, task_idx=task)

    class Walker2DV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("walker2d-medium-v2")
            super().__init__(env=env, task_idx=task)

    set_seed(seed)

    policy = None
    value_function = None
    if pretrained_dir is not None:
        snapshotter = Snapshotter()
        data = snapshotter.load(pretrained_dir, itr=pretrained_epoch)
        policy = data["algo"].policy
        value_function = data["algo"].value_function

    trainer = Trainer(ctxt)
    if env_name in "HopperV2":
        env_class = HopperV2
    elif env_name in "HalfCheetahV2":
        env_class = HalfCheetahV2
    elif env_name in "Walker2DV2":
        env_class = Walker2DV2
    elif env_name in "HalfCheetahVelEnv":
        env_class = HalfCheetahVelEnv
    else:
        raise NotImplementedError("Only HopperV2/HalfCheetahV2/Walker2DV2 accepted")
    print(f"Using env {env_class}")

    tasks = task_sampler.SetTaskSampler(
        env_class,
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

    # meta_evaluator = OnlineMetaEvaluator(
    #     test_task_sampler=tasks,
    #     n_test_tasks=1,
    #     n_test_episodes=1,
    #     prefix="MetaTestAdapt",
    #     worker_class=RL2Worker,
    #     worker_args=dict(n_episodes_per_trial=20),
    # )
    meta_evaluator = AdaptiveMDPEvaluator(
        eval_env=env_class(), n_eval_episodes=20, device="cuda"
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


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="half_cheetah")
def hydra_wrapper(cfg: TrainConfig):
    import pprint

    from hydra.utils import get_original_cwd
    import omegaconf

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    pprint.pprint(cfg_dict)

    if cfg.use_wandb:  # Initialise WandB
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config=cfg_dict,
            name=cfg.wandb_run_name,
            # monitor_gym=cfg.monitor_gym,
            save_code=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    return hydra.utils.call(cfg.main_config)


if __name__ == "__main__":
    hydra_wrapper()
    # transformer_ppo_halfcheetah()
