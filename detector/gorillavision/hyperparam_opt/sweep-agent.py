import wandb
with open('./keyfile') as f:
    key = f.read()
    wandb.login(key=key)
from sweep_main import main

sweep_id = "3ys0xxy7"
wandb.agent(sweep_id=sweep_id, project="triplet-approach", function=main)