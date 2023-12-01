"""Evaluator which tests Meta-RL algorithms on test environments."""

import collections

from dowel import logger, tabular
import numpy as np
import torch
import wandb

from garage import AugmentedEpisodeBatch, EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        test_task_sampler (TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_eps (int): Number of episodes to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_episodes (int): Number of episodes to use for each adapted
            policy. The adapted policy should forget previous episodes when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.

    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        *,
        test_task_sampler,
        n_exploration_eps=10,
        n_test_tasks=None,
        n_test_episodes=1,
        prefix="MetaTest",
        test_task_names=None,
        worker_class=DefaultWorker,
        worker_args=None,
    ):
        self._test_task_sampler = test_task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._n_exploration_eps = n_exploration_eps
        self._eval_itr = 0
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None
        self._max_episode_length = None

    def evaluate(self, algo, test_episodes_per_task=None):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        logger.log("Sampling for adapation and meta-testing...")
        env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        if self._test_sampler is None:
            env = env_updates[0]()
            self._max_episode_length = env.spec.max_episode_length
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(
                    seed=get_seed(),
                    max_episode_length=self._max_episode_length,
                    n_workers=1,
                    worker_class=self._worker_class,
                    worker_args=self._worker_args,
                ),
                agents=algo.get_exploration_policy(),
                envs=env,
            )
        for env_up in env_updates:
            policy = algo.get_exploration_policy()
            eps = EpisodeBatch.concatenate(
                *[
                    self._test_sampler.obtain_samples(self._eval_itr, 1, policy, env_up)
                    for _ in range(self._n_exploration_eps)
                ]
            )
            adapted_policy = algo.adapt_policy(policy, eps)
            adapted_eps = self._test_sampler.obtain_samples(
                self._eval_itr,
                test_episodes_per_task * self._max_episode_length,
                adapted_policy,
            )
            adapted_episodes.append(adapted_eps)
        logger.log("Finished meta-testing...")

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + "/" if self._prefix else ""):
            log_multitask_performance(
                self._eval_itr,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, "discount", 1.0),
                name_map=name_map,
            )
        self._eval_itr += 1


class OnlineMetaEvaluator:
    """Evaluates Meta-RL algorithms with online adaptation on test environments.

    Args:
        test_task_sampler (TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_eps (int): Number of episodes to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_episodes (int): Number of episodes to use for each adapted
            policy. The adapted policy should forget previous episodes when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.

    """

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        *,
        test_task_sampler,
        n_test_tasks=30,
        n_test_episodes=5,
        prefix="MetaTest",
        test_task_names=None,
        worker_class=DefaultWorker,
        worker_args=None,
    ):
        self._test_task_sampler = test_task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._eval_itr = 0
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._episodes_per_trial = (
            worker_args["n_episodes_per_trial"]
            if "n_episodes_per_trial" in worker_args
            else 1
        )
        self._test_sampler = None
        self._max_episode_length = None

    def evaluate(self, algo, test_episodes_per_task=None):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        logger.log("Sampling for adapation and meta-testing...")
        env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        if self._test_sampler is None:
            env = env_updates[0]()
            self._max_episode_length = env.spec.max_episode_length
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(
                    seed=get_seed(),
                    max_episode_length=self._max_episode_length,
                    n_workers=self._n_test_tasks,
                    worker_class=self._worker_class,
                    worker_args=self._worker_args,
                ),
                agents=algo.get_exploration_policy(),
                envs=env_updates,
            )

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        if wandb.run is not None:
            wandb.log({"max_episode_length": self._max_episode_length})
            wandb.log({"n_test_tasks": self._n_test_tasks})
            wandb.log({"n_test_episodes": self._n_test_episodes})

        eps = self._test_sampler.obtain_samples(
            self._eval_itr,
            self._max_episode_length
            * self._n_test_episodes
            * self._episodes_per_trial
            * self._n_test_tasks,
            agent_update=algo.get_exploration_policy(),
            env_update=env_updates,
            deterministic=True,
        )

        episodes_by_order = self._cluster_by_episode_number(eps)

        for i in range(self._episodes_per_trial):
            with tabular.prefix(
                self._prefix + "_" + str(i) + "/" if self._prefix else str(i)
            ):
                log_multitask_performance(
                    self._eval_itr,
                    AugmentedEpisodeBatch.concatenate(*episodes_by_order[i]),
                    getattr(algo, "discount", 1.0),
                    name_map=name_map,
                )

        self._eval_itr += 1
        logger.log("Finished meta-testing...")

    def _cluster_by_episode_number(self, episodes):
        episode_idx = collections.defaultdict(list)
        episode_list = episodes.split()
        for episode_num in range(self._episodes_per_trial):
            for task_num in range(self._n_test_tasks):
                episode_idx[episode_num].append(
                    episode_list[self._episodes_per_trial * task_num + episode_num]
                )
        return episode_idx


class AdaptiveMDPEvaluator:
    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 20,
        seq_len: int = 10,
        device: str = "cuda",
    ):
        self.eval_env = eval_env
        self.num_eval_episodes = n_eval_episodes
        self.seq_len = seq_len
        self.device = device
        self.res_agent = True
        self.deterministic = True
        # self._test_task_sampler = test_task_sampler
        # self._worker_class = worker_class
        # if worker_args is None:
        #     self._worker_args = {}
        # else:
        #     self._worker_args = worker_args
        # if n_test_tasks is None:
        #     n_test_tasks = test_task_sampler.n_tasks
        # self._n_test_tasks = n_test_tasks
        # self._n_test_episodes = n_test_episodes
        # self._eval_itr = 0
        # self._prefix = prefix
        # self._test_task_names = test_task_names
        # self._episodes_per_trial = (
        #     worker_args["n_episodes_per_trial"]
        #     if "n_episodes_per_trial" in worker_args
        #     else 1
        # )
        # self._test_sampler = None
        # self._max_episode_length = None

    @torch.no_grad()
    # def evaluate(self, res_agent:bool=True, deterministic:bool=True, num_eval:int=20):
    def evaluate(self, algo, test_episodes_per_task=None):
        logger.log("Starting adaptive MDP evaluator...")
        rewards = np.zeros(self.num_eval_episodes)
        for n in range(self.num_eval_episodes):
            obs, done = self.eval_env.reset(), False
            ep_reward = 0
            # used for saving episode to the buffer
            ep_states = [obs]
            ep_actions = []
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
                # action of offline RL agent
                # with torch.no_grad():
                #     offline_act, _ = algo.policy.actforward(
                #         obs_tensor, True
                #     )  # offline policy action
                # first step, no context encoder to be used
                breakpoint()
                if len(ep_actions) < self.seq_len or not self.res_agent:
                    # algo.get_exploration_policy()
                    action, agent_info = algo.get_action(
                        obs_tensor, deterministic=self.deterministic
                    )
                    if self.deterministic and "mean" in agent_info:
                        action = agent_info["mean"]
                else:
                    raise NotImplementedError("res_agent=False only")
                action = action.cpu().numpy()
                obs, reward, done, info = self.eval_env.step(action)
                ep_actions.append(action)
                ep_states.append(obs)
                ep_reward += reward
            rewards[n] = ep_reward
        ep_reward = self.eval_env.get_normalized_score(np.mean(rewards)) * 100
        std = self.eval_env.get_normalized_score(np.std(rewards)) * 100
        if wandb.run is not None:
            metrics = {
                "AdaptiveMDP/AverageReturn": ep_reward,
                "AdaptiveMDP/StdReturn": std,
                "AdaptiveMDP/NumEpisodes": n + 1,
                "AdaptiveMDP/LastEpisodeLength": len(ep_actions),
            }
            wandb.log(metrics)

        logger.log("Finished adaptive MDP evaluator.")
        return ep_reward, std
