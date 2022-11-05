import functools
import types

from absl.testing import absltest
from acme import types
import jax
import numpy as onp
import numpy.testing as np_test

from otr.agents.otil import rewarder as otil_rewarder


class OTILRewarderTest(absltest.TestCase):

  def test_rewarder(self):
    num_expert_demos = 2
    episode_length = 10
    obs_size = 3
    expert_observations = jax.random.normal(
        jax.random.PRNGKey(0), (num_expert_demos, episode_length, obs_size))
    expert_demos = []
    # Create expert demonstrations
    for n in range(num_expert_demos):
      traj = []
      for t in range(episode_length):
        traj.append(types.Transition(expert_observations[n, t], (), (), (), ()))
      expert_demos.append(traj)
    # Create agent demonstrations
    agent_demo = []
    agent_observations = jax.random.normal(
        jax.random.PRNGKey(0), (episode_length, obs_size))
    for t in range(episode_length):
      agent_demo.append(types.Transition(agent_observations[t], (), (), (), ()))
    # Compute rewards with the rewarder
    rewarder = otil_rewarder.OTILRewarder(expert_demos, episode_length)
    rewards = (rewarder.compute_offline_rewards(agent_demo)).tolist()
    self.assertLen(rewards, episode_length)

  def test_zero_rewards_for_expert_demo(self):
    num_expert_demos = 1
    episode_length = 10
    obs_size = 3
    expert_observations = jax.random.normal(
        jax.random.PRNGKey(0), (num_expert_demos, episode_length, obs_size))
    expert_demos = []
    # Create expert demonstrations
    for n in range(num_expert_demos):
      traj = []
      for t in range(episode_length):
        traj.append(types.Transition(expert_observations[n, t], (), (), (), ()))
      expert_demos.append(traj)
    # Create agent demonstrations
    agent_demos = expert_demos[0]
    # Compute rewards with the rewarder
    rewarder = otil_rewarder.OTILRewarder(
        expert_demos,
        episode_length,
        squashing_fn=functools.partial(
            otil_rewarder.squashing_linear, alpha=1.))
    rewards = (rewarder.compute_offline_rewards(agent_demos))
    np_test.assert_allclose(rewards, onp.zeros_like(rewards), atol=1e-3)


if __name__ == '__main__':
  absltest.main()
