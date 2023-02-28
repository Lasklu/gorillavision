<h1 align="center">GorillaVision</h1>

<h2 align="center" style=font-size:200px>An open-set re-identification system for gorillas in the wild</h2>

<a name="overview"></a>
## Overview

We present a system for open-set gorilla re-identification in the wild. Our system follows a two stage approach in which gorilla faces are detected with a YOLOv7 detector in the first stage, and are classified with our GorillaVision model in the second stage. We implement our classification model based on the VisionTransformer that is optimized with Triplet Loss and that computes embeddings of gorilla faces. As in many face-identification tasks, the embeddings are then used, to provide a similarity measure between the individual gorillas. Classification is then performed on these embeddings with a k-nearest neighbors algorithm. For a closed-set scenario, our approach slightly outperforms the state-of-the art YOLO detector. In the open-set scenario, our model is also able to deliver high quality results with an accuracy in the range of 60 to 80\% depending on the quality of the dataset. Given that we have many individuals with at least 6 images each, our approach achieves 89\% top-5 accuracy.

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
