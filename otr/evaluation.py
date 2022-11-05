import time
from typing import Optional

import acme
from acme import core
from acme.utils import counting
from acme.utils import loggers
import dm_env


class D4RLEvalLoop(core.Worker):

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: acme.Actor,
      label: str = "evaluation",
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)

  def run(self, num_episodes: int):  # pylint: disable=arguments-differ
    # Update actor once at the start
    self._actor.update(wait=True)
    total_episode_return = 0.0
    total_episode_steps = 0
    start_time = time.time()

    for _ in range(num_episodes):
      timestep = self._environment.reset()
      self._actor.observe_first(timestep)
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        timestep = self._environment.step(action)
        self._actor.observe(action, timestep)
        total_episode_steps += 1
        total_episode_return += timestep.reward

    steps_per_second = total_episode_steps / (time.time() - start_time)
    counts = self._counter.increment(
        steps=total_episode_steps, episodes=num_episodes)
    average_episode_return = total_episode_return / num_episodes
    average_episode_steps = total_episode_steps / num_episodes
    average_normalized_return = self._environment.get_normalized_score(
        average_episode_return)
    result = {
        "average_episode_return": average_episode_return,
        "average_normalized_return": average_normalized_return,
        "average_episode_length": average_episode_steps,
        "steps_per_second": steps_per_second,
    }
    result.update(counts)
    self._logger.write(result)
