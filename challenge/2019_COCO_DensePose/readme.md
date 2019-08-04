# COCO 2019 DensePose Task

![DensePose Splash Image](http://cocodataset.org/images/densepose-splash.png)

## Overview

The COCO DensePose Task requires dense estimation of human pose in challenging,
uncontrolled conditions. The DensePose task involves simultaneously detecting
people, segmenting their bodies and mapping all image pixels that belong to a
human body to the 3D surface of the body. For full details of this task please
see the [DensePose evaluation](evaluation.md) page.

This task is part of the
[COCO+Mapillary Joint Recognition Challenge Workshop](http://cocodataset.org/workshop/coco-mapillary-iccv-2019.html)
at ICCV 2019. For further details about the joint workshop, as well as **new rules regarding technical reports and awards**, please
visit the workshop page. 
Please also see the related COCO
[detection](http://cocodataset.org/workshop/coco-mapillary-iccv-2019.html#coco-detection),
[panoptic](http://cocodataset.org/workshop/coco-mapillary-iccv-2019.html#coco-panoptic)
and
[keypoints](http://cocodataset.org/workshop/coco-mapillary-iccv-2019.html#coco-keypoints)
tasks.

The COCO train, validation, and test sets, containing more than 39,000 images
and 56,000 person instances labeled with DensePose annotations are available
for [download](http://cocodataset.org/#download).
Annotations on train (
[train 1](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json),
[train 2](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_valminusminival.json)
) and [val](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json)
with over 48,000 people are publicly available.
[Test set](https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_test.json)
with the list of images is also available for download.

Evaluation server for the 2019 task is
[open](https://competitions.codalab.org/competitions/20660).

## Dates

[]() | []()
---- | -----
**October 4, 2019** | Submission deadline (23:59 PST)
October 11, 2019  | Technical report submission deadline
October 18, 2019  | Challenge winners notified
October 27, 2019  | Winners present at ICCV 2019 Workshop

## Organizers

Vasil Khalidov (Facebook AI Research)

Natalia Neverova (Facebook AI Research)

Riza Alp Güler (Imperial College London / Ariel AI)

Iasonas Kokkinos (UCL / Ariel AI)

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
[evaluation server](https://competitions.codalab.org/competitions/20660).
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
For additional questions, please contact vkhalidov@fb.com and nneverova@fb.com.

