import wandb
with open('./keyfile') as f:
    key = f.read()
    wandb.login(key=key)
from sweep_main import main

sweep_id = "sib24e5r"
wandb.agent(sweep_id=sweep_id, project="triplet-approach", function=main)