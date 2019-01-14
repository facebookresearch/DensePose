#!/bin/bash
mkdir DensePose_COCO
cd DensePose_COCO
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_valminusminival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_test.json
