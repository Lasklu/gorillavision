import os
import yaml
from train import train
import wandb
import numpy as np
from utils.logger import logger
from utils.dataset_utils import load_data
from utils.dataset_statistics import compute_statistics
from model.triplet import TripletLoss
from score import score
from create_db import create_db

sweep_configuration = None
with open("sweep.yaml", "r") as stream:
    sweep_configuration = yaml.safe_load(stream)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="triplet-approach")