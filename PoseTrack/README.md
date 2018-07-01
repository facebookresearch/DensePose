#  DensePose-PoseTrack
We introduce the DensePose-Posetrack dataset, which consists of videos of multiple personcontain rapid motions, occlusions and scale variation which leads to a very challenging correspondence task. DensePose-PoseTrack will be a part of the [ECCV 2018 - POSETRACK CHALLENGE](https://posetrack.net/workshops/eccv2018/).

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1fed2Xvy2G6t4V_ICsEJIm-PaJ8o-e0Ws" width="700px" />
</div>

Please first follow the [INSTALL.md](https://github.com/facebookresearch/DensePose/blob/master/INSTALL.md) and [GETTING_STARTED.md](https://github.com/facebookresearch/DensePose/blob/master/GETTING_STARTED.md), to install and run the DensePose inference and training. This documents follows by instructions to download and evaluate on the DensePose-PoseTrack dataset.

### Fetch DensePose-PoseTrack dataset
Note that the DensePose-PoseTrack dataset is distributed under [NonCommercial Creative Commons](https://creativecommons.org/licenses/by-nc/2.0/) license.
To downoad, run:
```
cd $DENSEPOSE/PoseTrack
bash get_DensePose_PoseTrack.sh
```
This script downloads *.json files that contains all annotations along with files that only contains annotations for images with densepose annotations. The latter is used during evaluation.


## Setting-up the PoseTrack dataset.

Create a symlink for the PoseTrack dataset in your `datasets/data` folder.
```
ln -s /path/to/posetrack $DENSEPOSE/detectron/datasets/data/posetrack
```
Create symlinks for the DensePose-PoseTrack annotations

```
ln -s $DENSEPOSE/PoseTrack/DensePose_PoseTrack/densepose_only_posetrack_train2017.json $DENSEPOSE/detectron/datasets/data/posetrack/
ln -s $DENSEPOSE/PoseTrack/DensePose_PoseTrack/densepose_only_posetrack_val2017.json $DENSEPOSE/detectron/datasets/data/posetrack/
ln -s $DENSEPOSE/PoseTrack/DensePose_PoseTrack/densepose_posetrack_test2017.json $DENSEPOSE/detectron/datasets/data/posetrack/
```
Your local PoseTrack dataset copy at `/path/to/posetrack` should have the following directory structure:

```
posetrack
|_ images
|  |_ <im-folder-1>
|  |_ ...
|  |_ <im-folder-N>.
|_ densepose_only_posetrack_train2017.json
|_ densepose_only_posetrack_val2017.json
|_ densepose_posetrack_test2017.json
```

### Evaluation on DensePose-PoseTrack dataset

To demonstrate the evaluation, we use a DensePose-RCNN with a ResNet-50 trunk that is trained on the DensePose-COCO dataset.
```
cd $DENSEPOSE
python2 tools/test_net.py \
    --cfg PoseTrack/configs/DensePose_ResNet50_FPN_s1x-e2e.yaml \
    TEST.WEIGHTS https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x-e2e.pkl \
    NUM_GPUS 1
```
The evaluation of this baseline network should yield `Bounding Box AP: 0.4438` and `DensePose AP: 0.2698`.
