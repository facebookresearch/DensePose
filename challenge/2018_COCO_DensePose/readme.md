# COCO 2018 DensePose Task

![DensePose Splash Image](http://cocodataset.org/images/densepose-splash.png)

## Overview

The COCO DensePose Task requires dense estimation of human pose in challenging,
uncontrolled conditions. The DensePose task involves simultaneously detecting
people, segmenting their bodies and mapping all image pixels that belong to a
human body to the 3D surface of the body. For full details of this task please
see the [DensePose evaluation](evaluation.md) page.

This task is part of the
[Joint COCO and Mapillary Recognition Challenge Workshop](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html)
at ECCV 2018. For further details about the joint workshop please
visit the workshop page. Please also see the related COCO
[detection](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html#coco-detection),
[panoptic](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html#coco-panoptic),
[keypoints](http://cocodataset.org/workshop/coco-mapillary-eccv-2018.html#coco-keypoints)
and [stuff](http://cocodataset.org/#stuff-2018) tasks.

The COCO train, validation, and test sets, containing more than 39,000 images
and 56,000 person instances labeled with DensePose annotations are available
for [download](http://cocodataset.org/#download).
Annotations on train (
[train 1](https://s3.amazonaws.com/densepose/densepose_coco_2014_train.json),
[train 2](https://s3.amazonaws.com/densepose/densepose_coco_2014_valminusminival.json)
) and [val](https://s3.amazonaws.com/densepose/densepose_coco_2014_minival.json)
with over 48,000 people are publicly available.
[Test set](https://s3.amazonaws.com/densepose/densepose_coco_2014_test.json)
with the list of images is also available for download.

Evaluation server for the 2018 task is
[open](https://competitions.codalab.org/competitions/19636).

## Dates

[]() | []()
---- | -----
**August 17, 2018** | Submission deadline (23:59 PST)
August 26, 2018   | Challenge winners notified
September 9, 2018 | Winners present at ECCV 2018 Workshop

## Organizers

Riza Alp Güler (INRIA, CentraleSupélec)

Natalia Neverova (Facebook AI Research)

Iasonas Kokkinos (Facebook AI Research)

## Task Guidelines

Participants are recommended but not restricted to train
their algorithms on COCO DensePose train and val sets.
The [download](http://cocodataset.org/#download) page has
links to all COCO data. When participating in this task,
please specify any and all external data used for training
in the "method description" when uploading results to the
evaluation server. A more thorough explanation of all these
details is available on the
[guidelines](http://cocodataset.org/#guidelines) page,
please be sure to review it carefully prior to participating.
Results in the [correct format](results_format.md) must be
[uploaded](upload.md) to the
[evaluation server](https://competitions.codalab.org/competitions/19636).
The [evaluation](evaluation.md) page lists detailed information
regarding how results will be evaluated. Challenge participants
with the most successful and innovative methods will be invited
to present at the workshop.

## Tools and Instructions

We provide extensive API support for the COCO images,
annotations, and evaluation code. To download the COCO DensePose API,
please visit our
[GitHub repository](https://github.com/facebookresearch/DensePose/).
Due to the large size of COCO and the complexity of this task,
the process of participating may not seem simple. To help, we provide
explanations and instructions for each step of the process:
[download](http://cocodataset.org/#download),
[data format](data_format.md),
[results format](results_format.md),
[upload](upload.md) and [evaluation](evaluation.md) pages.
For additional questions, please contact info@cocodataset.org.



