import wandb
from initsweep import main

sweep_id = "iehivaw7"
wandb.agent(sweep_id=sweep_id, function=main)