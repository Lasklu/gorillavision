import yaml
import wandb

sweep_configuration = None
with open("sweep.yaml", "r") as stream:
    sweep_configuration = yaml.safe_load(stream)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="triplet-approach")