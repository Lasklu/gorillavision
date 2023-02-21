<h1 align="center">GorillaVision</p>

<h2 align="center" style=font-size:200px>An open-set re-identification system for gorillas in the wild</h1>

<a name="overview"></a>
## Overview

ToDo Add abstract here

<a name="running"></a>
### Running the Application

1. Build a docker image from the `dockerfile` in the `embeddings-approach` folder.
2. Create an empty folder where the models should be saved
3. Ensure that your data folder contains the following subfolder: train, database and eval. For an explaination of these, please check our corresponding paper in the section dataset splits.
4. Adapt the `configs/config.json` file according to your needs, especially ensure to provide the correct dataset path.
5. Run the docker container with mounting the repository, the data folder and the model_out folder. An example of this could look like this:
```
docker run -v  /home/mydatafolder/:/data -v /home/models/:/models -v /gorilla-reidentification/embedding-approach:/gorilla-reidentification/embedding-approach --gpus device=1 --ipc="host" -it gorilla_triplet python3 eval_pipeline.py
```
This will start training and evaluation based on the dataset you provided.

<a name="contributors"></a>
### Contributors ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/rohansaw"><img src="https://avatars.githubusercontent.com/u/49531442?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Rohan Sawahn</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Lasklu"><img src="https://avatars.githubusercontent.com/u/49564344?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lukas Laskowski</b></sub></a><br /></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
