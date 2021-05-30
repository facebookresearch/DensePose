# PoseTrack DensePose Evaluation

This page describes the DensePose evaluation metrics. The
evaluation code provided here can be used to obtain results on the publicly
available PoseTrack DensePose validation set. It computes multiple metrics
described below. To obtain results on the PoseTrack DensePose test set,
for which ground-truth annotations are hidden, generated results must be
uploaded to the
[evaluation server](https://competitions.codalab.org/competitions/19650).
The exact same evaluation code,
described below, is used to evaluate results on the test set.

## Evaluation Overview

The multi-person PoseTrack DensePose task involves simultaneous person detection,
segmentation and
estimation of correspondences between image pixels that belong to a human body
and a template 3D model through time.
DensePose evaluation mimics the evaluation metrics
used for [object detection](http://cocodataset.org/#detection-eval) and
[keypoint estimation](http://cocodataset.org/#keypoints-eval) in the COCO
challenge, namely average precision (AP) and average recall (AR) and their
variants.

At the heart of these metrics is a similarity measure between ground truth
objects and predicted objects. In the case of object detection,
*Intersection over Union* (IoU) serves as this similarity measure (for both
boxes and segments). Thesholding the IoU defines matches between the ground
truth and predicted objects and allows computing precision-recall curves.
In the case of keypoint detection *Object Keypoint Similarity* (OKS) is used.

To adopt AP/AR for dense correspondence, we define an analogous similarity
measure called *Geodesic Point Similarity* (GPS) which plays the same role
as IoU for object detection and OKS for keypoint estimation. 

## Geodesic Point Similarity

The geodesic point similarity (GPS) is based on geodesic distances on the template mesh between the collected groundtruth points and estimated surface coordinates for the same image points as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{GPS}&space;=&space;\frac{1}{|P|}\sum_{p_i&space;\in&space;P}\exp\left&space;(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right)," target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\text{GPS}&space;=&space;\frac{1}{|P|}\sum_{p_i&space;\in&space;P}\exp\left&space;(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right),"
title="https://www.codecogs.com/eqnedit.php?latex=\text{GPS} = \frac{1}{|P|}\sum_{p_i \in P}\exp\left(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right)," /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=&space;d(\hat{p}_i,p_i)&space;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?&space;d(\hat{p}_i,p_i)&space;" title="https://www.codecogs.com/eqnedit.php?latex=d(\hat{p}_i,p_i)" /></a> is the geodesic distance between estimated
(<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{p}_i" target="_blank"> <img src="https://latex.codecogs.com/gif.latex?\hat{p}_i" title="https://www.codecogs.com/eqnedit.php?latex=\hat{p}_i" /></a>) and groundtruth
(<a href="https://www.codecogs.com/eqnedit.php?latex=p_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i" title="https://www.codecogs.com/eqnedit.php?latex=p_i" /></a>)
human body surface points and
<a href="https://www.codecogs.com/eqnedit.php?latex=\kappa(p_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\kappa(p_i)" title="https://www.codecogs.com/eqnedit.php?latex=\kappa(p_i)" /></a>
is a per-part normalization factor, defined as the mean geodesic distance between points on the part.

## Metrics

The following metrics are used to characterize the performance of a dense pose
estimation algorithm on COCO:

*Average Precision*
```
AP       % AP averaged over GPS values 0.5 : 0.05 : 0.95 (primary challenge metric)
AP-50    % AP at GPS=0.5  (loose metric)
AP-75    % AP at GPS=0.75 (strict metric)
AP-m     % AP for medium detections: 32² < area < 96²
AP-l     % AP for large detections:  area > 96²
```

## Evaluation Code

Evaluation code is available on the
[DensePose](https://github.com/facebookresearch/DensePose/) github,
see [densepose_cocoeval.py](https://github.com/facebookresearch/DensePose/blob/master/detectron/datasets/densepose_cocoeval.py).
Before running the evaluation code, please prepare your results in the format
described on the [results](results_format.md) format page.
The geodesic distances are pre-computed on a subsampled version of the SMPL
model to allow faster evaluation. Geodesic distances are computed after
finding the closest vertices to the estimated UV values in the subsampled mesh.

