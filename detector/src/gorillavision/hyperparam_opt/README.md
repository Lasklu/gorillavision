To run the wandb hyperparam optimization follow the following steps:

0. Clone this repo
1. Paste the Wandb API-Key into `/hyperparam_opt/keyfile`
2. Build the docker container that is in `/dockerfile`
3. Run the dockercontainer with mounting the repository and the dataset, so that the dataset is mounted to /data/<datasetname>
4. Run sweep.sh