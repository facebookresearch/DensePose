# DensePose Model Zoo

## RPN Files
Herein, we provide RPN files for DensePose-COCO dataset `train`, `minival` and `valminusminival` partitions.

The RPN results are obtained using the models provided in the Detectron model-zoo. For performance measures please refer to [`this file`](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#person-specific-rpn-baselines).

**X-101-32x8d-FPN:** [`[train]`](https://s3.amazonaws.com/densepose/DensePose-RPN-train_X-101-32x8d-FPN.pkl) [`[minival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-minival_X-101-32x8d-FPN.pkl) [`[valminusminival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-valminusminival_X-101-32x8d-FPN.pkl)

**R-50-FPN:** [`[train]`](https://s3.amazonaws.com/densepose/DensePose-RPN-train_fpn_resnet50.pkl) [`[minival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-minival_fpn_resnet50.pkl) [`[valminusminival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-valminusminival_fpn_resnet50.pkl)

## DensePose-RCNN Models

| Model | AP  |  AP50 | AP75  | APm  |APl |
|-----|-----|---    |---    |---   |--- |
| [`ResNet50_FPN_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x.pkl)| 0.4748 |0.8368|0.4820 |0.4262|0.4948|
| [`ResNet50_FPN_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x-e2e.pkl)|0.4892 |0.8490|0.5078| 0.4384|0.5059|
| [`ResNet101_FPN_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x.pkl)|0.4978| 0.8521|0.5276 |0.4373|0.5164|
| [`ResNet101_FPN_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl)|0.5147 |0.8660|0.5601 |0.4716|0.5291|
| [`ResNet101_FPN_32x8d_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_32x8d_s1x.pkl)|0.5095 | 0.8590|0.5381 |0.4605|0.5272|
| [`ResNet101_FPN_32x8d_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pkl)|0.5554 | 0.8908|0.6080 |0.5067|0.5676|

Please note that due to the new per-part normalization the AP numbers do not match those reported in the paper, which are obtained with global normalization factor `K = 0.255`.

## Models with Multiple Heads

We provide an example of a
[configuration file](configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml)
that performs multiple tasks
using the same backbone architecture (ResNet-50) and containing several
heads for dense pose, mask and keypoints estimation. We note that this
example is provided purely for illustrative purposes and the performance
of the model is not tuned, so it occurs to be inferior compared to single
head models for each individual task.

| Task | AP  |  AP50 | AP75  | APm  |APl |
|-----|-----|---    |---    |---   |--- |
| mask | 0.4708 | 0.8021 | 0.5002 | 0.4159 | 0.6227 |
| keypoint | 0.5871 | 0.8452 | 0.6309 | 0.4611 | 0.6931 |
| densepose | 0.4726 | 0.8480 | 0.4750 | 0.3851 | 0.4929 |

([config](configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml),
[model](https://s3.amazonaws.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl),
[md5](https://s3.amazonaws.com/densepose/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.md5))
