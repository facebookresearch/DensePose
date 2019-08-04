# DensePose Evaluation

This page describes the DensePose evaluation metrics used by COCO. The
evaluation code provided here can be used to obtain results on the publicly
available COCO DensePose validation set. It computes multiple metrics
described below. To obtain results on the COCO DensePose test set, for which
ground-truth annotations are hidden, generated results must be uploaded to
the evaluation server. The exact same evaluation code, described below, is
used to evaluate results on the test set.

**Please note the changes in the evaluation metric of the 2019 challenge compared to 2018 
(see description below).**

## Evaluation Overview

The multi-person DensePose task involves simultaneous person detection and
estimation of correspondences between image pixels that belong to a human body
and a template 3D model. DensePose evaluation mimics the evaluation metrics
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

To adopt AP/AR for dense correspondence, an analogous similarity
measure called *Geodesic Point Similarity* (GPS) has been introduced,
which plays the same role as IoU for object detection and OKS for keypoint estimation. 

## Geodesic Point Similarity

The geodesic point similarity (GPS) is based on geodesic distances on the template mesh
between the collected ground truth points and estimated surface coordinates for the same image points as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{GPS}&space;=&space;\frac{1}{|P|}\sum_{p_i&space;\in&space;P}\exp\left&space;(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right)," target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\text{GPS}&space;=&space;\frac{1}{|P|}\sum_{p_i&space;\in&space;P}\exp\left&space;(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right),"
title="https://www.codecogs.com/eqnedit.php?latex=\text{GPS} = \frac{1}{|P|}\sum_{p_i \in P}\exp\left(\frac{-{d(\hat{p}_i,p_i)}^2}{2\kappa(p_i)^2}\right)," /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=&space;d(\hat{p}_i,p_i)&space;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?&space;d(\hat{p}_i,p_i)&space;" title="https://www.codecogs.com/eqnedit.php?latex=d(\hat{p}_i,p_i)" /></a> is the geodesic distance between estimated
(<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{p}_i" target="_blank"> <img src="https://latex.codecogs.com/gif.latex?\hat{p}_i" title="https://www.codecogs.com/eqnedit.php?latex=\hat{p}_i" /></a>) and groundtruth
(<a href="https://www.codecogs.com/eqnedit.php?latex=p_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i" title="https://www.codecogs.com/eqnedit.php?latex=p_i" /></a>)
human body surface points and
<a href="https://www.codecogs.com/eqnedit.php?latex=\kappa(p_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\kappa(p_i)" title="https://www.codecogs.com/eqnedit.php?latex=\kappa(p_i)" /></a>
is a per-part normalization factor, defined as the mean geodesic distance between points on the part. **Please note that due to the new per-part normalization the AP numbers do not match those reported in the paper, which are obtained via fixed K = 0.255.**

This formulation has a limitation that it is estimated on a set of predefined
annotated points and therefore does not penalize spurious detections (false positives).
As a result, the metric erroneously favors predictions with all pixels classified
as foreground. To account for this, we introduce an additional multiplicative term
corresponding to the intersection over union (IoU) between the ground truth and the
predicted foreground masks to obtain an improved *masked-GPS*.


## Masked Geodesic Point Similarity

The masked geodesic point similarity (GPSm) is calculated as

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{GPS}^m=\sqrt{\text{GPS}\cdot\mathcal{I}},\quad\text{with}\quad\mathcal{I}=\frac{\mathcal{M}\cap\hat{\mathcal{M}}}{\mathcal{M}\cup\hat{\mathcal{M}}}," target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\text{GPS}^m=\sqrt{\text{GPS}\cdot\mathcal{I}},\quad\text{with}\quad\mathcal{I}=\frac{\mathcal{M}\cap\hat{\mathcal{M}}}{\mathcal{M}\cup\hat{\mathcal{M}}},"
title="https://www.codecogs.com/eqnedit.php?latex=\text{GPS}^m=\sqrt{\text{GPS}\cdot\mathcal{I}},\quad\text{with}\quad\mathcal{I}=\frac{\mathcal{M}\cap\hat{\mathcal{M}}}{\mathcal{M}\cup\hat{\mathcal{M}}}," /></a>

where GPS is the geodesic point similarity metric value and
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{I}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\mathcal{I}"
title="https://www.codecogs.com/eqnedit.php?latex=\mathcal{I}" /></a>
is the intersection over union between the ground truth (
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{M}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\mathcal{M}"
title="https://www.codecogs.com/eqnedit.php?latex=\mathcal{M}" /></a>
)
and the predicted (
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\mathcal{M}}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\hat{\mathcal{M}}"
title="https://www.codecogs.com/eqnedit.php?latex=\hat{\mathcal{M}}" /></a>
)
foreground masks.

## Metrics

The following metrics are used to characterize the performance of a dense pose
estimation algorithm on COCO:

*Average Precision*
```
AP       % AP averaged over GPSm values 0.5 : 0.05 : 0.95 (primary challenge metric)
AP-50    % AP at GPSm=0.5  (loose metric)
AP-75    % AP at GPSm=0.75 (strict metric)
AP-m     % AP for medium detections: 32² < area < 96²
AP-l     % AP for large detections:  area > 96²
```

## Evaluation Code

Evaluation code is available on the
[DensePose](https://github.com/facebookresearch/DensePose/) github,
see [densepose_cocoeval.py](https://github.com/facebookresearch/DensePose/blob/master/challenge/2019_COCO_DensePose/densepose_cocoeval.py).
Before running the evaluation code, please prepare your results in the format
described on the [results](results_format.md) format page.
The geodesic distances are pre-computed on a subsampled version of the SMPL
model to allow faster evaluation. Geodesic distances are computed after
finding the closest vertices to the estimated UV values in the subsampled mesh.

