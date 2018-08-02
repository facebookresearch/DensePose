#!/bin/bash
mkdir DensePose_COCO
cd DensePose_COCO
wget https://s3.amazonaws.com/densepose/densepose_coco_2014_train.json
wget https://s3.amazonaws.com/densepose/densepose_coco_2014_valminusminival.json
wget https://s3.amazonaws.com/densepose/densepose_coco_2014_minival.json
wget https://s3.amazonaws.com/densepose/densepose_coco_2014_test.json
