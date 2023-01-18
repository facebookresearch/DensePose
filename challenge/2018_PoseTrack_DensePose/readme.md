# ECCV 2018 PoseTrack DensePose Task

![PoseTrack DensePose Splash Image](https://posetrack.net/workshops/eccv2018/assets/images/densepose-posetrack_examples.jpg)

## Overview

The PoseTrack DensePose Task requires dense estimation of human pose through time
in challenging, uncontrolled conditions. The task involves processing video frames
to simultaneously detect people, segment their bodies and map all image pixels
that belong to a human body to the 3D surface of the body. For full details on
this task please see the [evaluation](evaluation.md) page.

This task is part of the
[PoseTrack Challenge Workshop](https://posetrack.net/workshops/eccv2018/)
at ECCV 2018. For further details about the workshop please
visit the workshop page. Please also see the related PoseTrack
[Articulated Human Pose Estimation and Tracking](https://posetrack.net/workshops/eccv2018/#challenges)
and 
[3D Human Pose Estimation](https://posetrack.net/workshops/eccv2018/#challenges)
tasks.

The PoseTrack DensePose train, validation, and test sets, containing more
than 5,000 images
and 27,000 person instances labeled with DensePose annotations are available
for [download](https://posetrack.net/users/download.php).
Annotations on
[train](https://www.dropbox.com/s/tpbaemzvlojo2iz/densepose_only_posetrack_train2017.json?dl=1)
and
[val](https://www.dropbox.com/s/43h43s0t3hkuogr/densepose_only_posetrack_val2017.json?dl=1)
with over 13,000 people are publicly available.

Evaluation server for the 2018 task is
[open](https://competitions.codalab.org/competitions/19650).

## Dates

[]() | []()
---- | -----
**August 18, 2018** | Submission deadline (23:59 PST)
September 2, 2018   | Challenge winners notified
September 8, 2018 | Winners present at ECCV 2018 Workshop

## Organizers

Riza Alp Güler (INRIA, CentraleSupélec)

Natalia Neverova (Facebook AI Research)

Iasonas Kokkinos (Facebook AI Research)

## Task Guidelines

Participants are recommended but not restricted to train
their algorithms on PoseTrack DensePose train and val sets.
The [download](https://posetrack.net/users/download.php) page has
links to the image data. When participating in this task,
please specify any and all external data used for training
in the "method description" when uploading results to the
evaluation server. **Listing external data used is mandatory.**
We emphasize that any form of **annotation or use of the test sets
for supervised or unsupervised training is strictly forbidden**.
A more thorough explanation of all these
details is available on the
[upload](upload.md) page,
please be sure to review it carefully prior to participating.
Results in the [correct format](results_format.md) must be
[uploaded](upload.md) to the
[evaluation server](https://competitions.codalab.org/competitions/19650).
The [evaluation](evaluation.md) page lists detailed information
regarding how results will be evaluated. Challenge participants
with the most successful and innovative methods will be invited
to present at the workshop.

## Tools and Instructions

We provide extensive API support for the images, annotations,
and evaluation code. To download the COCO DensePose API,
please visit our
[GitHub repository](https://github.com/facebookresearch/DensePose/).
Due to the complexity of this task, the process of participating
may not seem simple. To help, we provide explanations and
instructions for each step of the process:
[download](https://posetrack.net/users/download.php),
[data format](data_format.md),
[results format](results_format.md),
[upload](upload.md) and [evaluation](evaluation.md) pages.

