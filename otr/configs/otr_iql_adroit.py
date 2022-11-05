from ml_collections import config_dict

_NUM_SEEDS = 10


def get_config():
  config = config_dict.ConfigDict()
  config.batch_size = 256
  config.max_steps = int(1e6)
  config.evaluate_every = int(1e4)
  config.evaluation_episodes = 10
  config.seed = 0
  config.use_dataset_reward = False
  config.wandb_project = 'otr'
  config.wandb_entity = None
  config.expert_dataset_name = 'hammer-cloned-v0'
  config.offline_dataset_name = 'hammer-cloned-v0'
  config.k = 10

  # OTIL config
  config.squashing_fn = 'exponential'
  config.alpha = 5.0
  config.beta = 5.0

  # IQL config
  config.opt_decay_schedule = "cosine"
  config.dropout_rate = 0.1
  config.actor_lr = 3e-4
  config.value_lr = 3e-4
  config.critic_lr = 3e-4
  config.hidden_dims = (256, 256)
  config.iql_kwargs = dict(
      discount=0.99,
      expectile=0.7,  # The actual tau for expectiles.
      temperature=0.5)
  config.log_to_wandb = False
  return config


def get_sweep(h):
  del h
  params = []
  datasets = [
      'pen-human-v0',
      'hammer-human-v0',
      'door-human-v0',
      'relocate-human-v0',
      'pen-cloned-v0',
      'hammer-cloned-v0',
      'door-cloned-v0',
      'relocate-cloned-v0',
  ]
  for seed in range(_NUM_SEEDS):
    for use_dataset_reward in [False]:
      for squash in ['exponential']:
        for dataset in datasets:
          for k in [1, 10]:
            params.append({
                'config.offline_dataset_name': dataset,
                # Dummy to disambiguate
                'config.expert_dataset_name': dataset,
                'config.squashing_fn': squash,
                'config.k': k,
                'config.seed': seed,
                'config.use_dataset_reward': use_dataset_reward
            })
  return params
