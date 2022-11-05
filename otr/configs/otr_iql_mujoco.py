from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict()
  # Batch size used for training
  config.batch_size = 256
  # Number of training steps
  config.max_steps = int(1e6)
  # Interval to run evaluation
  config.evaluate_every = int(1e4)
  # Number of evaluation episodes to run
  config.evaluation_episodes = 10
  # Random seed for the experiment
  config.seed = 0
  # If true, use the reward from the original dataset
  # If false, use the reward computed by OTR
  config.use_dataset_reward = False
  # If true, log to wandb
  config.log_to_wandb = False
  # Wandb logging configuration
  config.wandb_project = 'otr'
  config.wandb_entity = None
  # D4RL dataset for picking the expert demonstration
  config.expert_dataset_name = 'hopper-medium-replay-v2'
  # No. of episodes to pick from the expert dataset to use
  config.k = 10
  # D4RL dataset to use as the unlabeled dataset
  config.offline_dataset_name = 'hopper-medium-replay-v2'

  # OTIL config
  # Squashing function to use
  config.squashing_fn = 'exponential'
  # Hyperparameters for the squashing function
  config.alpha = 5.0
  config.beta = 5.0

  # IQL config
  # These hyperparameters follow from the original IQL implementation.
  config.opt_decay_schedule = "cosine"
  config.actor_lr = 3e-4
  config.dropout_rate = None
  config.value_lr = 3e-4
  config.critic_lr = 3e-4
  config.hidden_dims = (256, 256)
  config.iql_kwargs = dict(
      discount=0.99,
      expectile=0.7,  # The actual tau for expectiles.
      temperature=3.0)
  return config


_NUM_SEEDS = 10

def get_sweep(h):
  del h
  params = []
  for seed in range(_NUM_SEEDS):
    for task in ['walker2d', 'hopper', 'halfcheetah']:
      for quality in ['medium', 'medium-replay', 'medium-expert']:
        for num_demos in [1, 10]:
          params.append({
              'config.expert_dataset_name': f'{task}-{quality}-v2',
              'config.k': num_demos,
              'config.offline_dataset_name': f'{task}-{quality}-v2',
              'config.seed': seed,
          })
  return params
