# DensePose Model Zoo

## RPN Files
Herein, we provide RPN files for DensePose-COCO dataset `train`, `minival` and `valminusminival` partitions.

The RPN results are obtained using the models provided in the Detectron model-zoo. For performance measures please refer to [`this file`](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#person-specific-rpn-baselines).

**X-101-32x8d-FPN:** [`[train]`](https://s3.amazonaws.com/densepose/DensePose-RPN-train_X-101-32x8d-FPN.pkl) [`[minival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-minival_X-101-32x8d-FPN.pkl) [`[valminusminival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-valminusminival_X-101-32x8d-FPN.pkl)

**R-50-FPN:** [`[train]`](https://s3.amazonaws.com/densepose/DensePose-RPN-train_fpn_resnet50.pkl) [`[minival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-minival_fpn_resnet50.pkl) [`[valminusminival]`](https://s3.amazonaws.com/densepose/DensePose-RPN-valminusminival_fpn_resnet50.pkl)

## DensePose-RCNN Models

| Model | AP  |  AP50 | AP75  | APm  |APl |
|-----|-----|---    |---    |---   |--- |
| [`ResNet50_FPN_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x.pkl)| 0.5119 |0.8538|0.5409 |0.4532|0.5322|
| [`ResNet50_FPN_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet50_FPN_s1x-e2e.pkl)|0.5221 |0.8556|0.5581| 0.4746|0.5422|
| [`ResNet101_FPN_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x.pkl)|0.5461| 0.8695|0.5828 |0.4871|0.5651|
| [`ResNet101_FPN_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl)|0.5598 |0.8754|0.6095 |0.5129|0.5744|
| [`ResNet101_FPN_32x8d_s1x`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_32x8d_s1x.pkl)|0.5584 | 0.8690|0.6029 |0.5074|0.5767|
| [`ResNet101_FPN_32x8d_s1x-e2e`](https://s3.amazonaws.com/densepose/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pkl)|0.6027 | 0.8989|0.6684 |0.5515|0.6150|
