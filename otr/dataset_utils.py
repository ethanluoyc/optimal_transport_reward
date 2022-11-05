from typing import Any, Iterator, Optional

from absl import logging
from acme import types
from acme import wrappers
import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tree


def get_d4rl_dataset(env):
  dataset = d4rl.qlearning_dataset(env)
  return types.Transition(
      observation=dataset["observations"],
      action=dataset["actions"],
      reward=dataset["rewards"],
      next_observation=dataset["next_observations"],
      discount=1.0 - dataset["terminals"].astype(np.float32),
  )


def make_environment(name: str, seed: Optional[int] = None):
  environment = gym.make(name)
  if seed is not None:
    environment.seed(seed)
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  environment = wrappers.CanonicalSpecWrapper(environment)
  return environment


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
  trajs = [[]]

  for i in tqdm.tqdm(range(len(observations))):
    trajs[-1].append(
        types.Transition(
            observation=observations[i],
            action=actions[i],
            reward=rewards[i],
            discount=masks[i],
            next_observation=next_observations[i]))
    if dones_float[i] == 1.0 and i + 1 < len(observations):
      trajs.append([])

  return trajs


def merge_trajectories(trajs):
  flat = []
  for traj in trajs:
    for transition in traj:
      flat.append(transition)
  return tree.map_structure(lambda *xs: np.stack(xs), *flat)


def qlearning_dataset_with_timeouts(env,
                                    dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
  if dataset is None:
    dataset = env.get_dataset(**kwargs)

  N = dataset['rewards'].shape[0]
  obs_ = []
  next_obs_ = []
  action_ = []
  reward_ = []
  done_ = []
  realdone_ = []
  if "infos/goal" in dataset:
    if not disable_goal:
      dataset["observations"] = np.concatenate(
          [dataset["observations"], dataset['infos/goal']], axis=1)
    else:
      pass
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)
      # dataset["observations"] = np.concatenate([
      #     dataset["observations"],
      #     np.zeros([dataset["observations"].shape[0], 2], dtype=np.float32)
      # ], axis=1)

  episode_step = 0
  for i in range(N - 1):
    obs = dataset['observations'][i]
    new_obs = dataset['observations'][i + 1]
    action = dataset['actions'][i]
    reward = dataset['rewards'][i]
    done_bool = bool(dataset['terminals'][i])
    realdone_bool = bool(dataset['terminals'][i])
    if "infos/goal" in dataset:
      final_timestep = True if (dataset['infos/goal'][i] !=
                                dataset['infos/goal'][i + 1]).any() else False
    else:
      final_timestep = dataset['timeouts'][i]

    if i < N - 1:
      done_bool += final_timestep

    if (not terminate_on_end) and final_timestep:
      # Skip this transition and don't apply terminals on the last step of an episode
      episode_step = 0
      continue
    if done_bool or final_timestep:
      episode_step = 0

    obs_.append(obs)
    next_obs_.append(new_obs)
    action_.append(action)
    reward_.append(reward)
    done_.append(done_bool)
    realdone_.append(realdone_bool)
    episode_step += 1

  return {
      'observations': np.array(obs_),
      'actions': np.array(action_),
      'next_observations': np.array(next_obs_),
      'rewards': np.array(reward_)[:],
      'terminals': np.array(done_)[:],
      'realterminals': np.array(realdone_)[:],
  }


def load_trajectories(name: str, fix_antmaze_timeout=True):
  env = gym.make(name)
  if "antmaze" in name and fix_antmaze_timeout:
    dataset = qlearning_dataset_with_timeouts(env)
  else:
    dataset = d4rl.qlearning_dataset(env)
  dones_float = np.zeros_like(dataset['rewards'])

  for i in range(len(dones_float) - 1):
    if np.linalg.norm(dataset['observations'][i + 1] -
                      dataset['next_observations'][i]
                     ) > 1e-6 or dataset['terminals'][i] == 1.0:
      dones_float[i] = 1
    else:
      dones_float[i] = 0
  dones_float[-1] = 1

  if 'realterminals' in dataset:
    # We updated terminals in the dataset, but continue using
    # the old terminals for consistency with original IQL.
    masks = 1.0 - dataset['realterminals'].astype(np.float32)
  else:
    masks = 1.0 - dataset['terminals'].astype(np.float32)
  traj = split_into_trajectories(
      observations=dataset['observations'].astype(np.float32),
      actions=dataset['actions'].astype(np.float32),
      rewards=dataset['rewards'].astype(np.float32),
      masks=masks,
      dones_float=dones_float.astype(np.float32),
      next_observations=dataset['next_observations'].astype(np.float32))
  return traj


def load_demonstrations(name: str, num_top_episodes: int = 10):
  """Load expert demonstrations."""
  # Load trajectories from the given dataset
  trajs = load_trajectories(name)
  if num_top_episodes < 0:
    logging.info("Loading the entire dataset as demonstrations")
    return trajs

  def compute_returns(traj):
    episode_return = 0
    for transition in traj:
      episode_return += transition.reward
    return episode_return

  # Sort by episode return
  trajs.sort(key=compute_returns)
  return trajs[-num_top_episodes:]


class JaxInMemorySampler(Iterator[Any]):

  def __init__(
      self,
      dataset,
      key: jnp.ndarray,
      batch_size: int,
  ):
    self._dataset_size = jax.tree_util.tree_leaves(dataset)[0].shape[0]
    self._jax_dataset = jax.tree_map(jax.device_put, dataset)

    def sample(data, key: jnp.ndarray):
      key1, key2 = jax.random.split(key)
      indices = jax.random.randint(
          key1, (batch_size,), minval=0, maxval=self._dataset_size)
      data_sample = jax.tree_map(lambda d: jnp.take(d, indices, axis=0), data)
      return data_sample, key2

    self._sample = jax.jit(lambda key: sample(self._jax_dataset, key))
    self._key = key

  def __next__(self) -> Any:
    data, self._key = self._sample(self._key)
    return data
