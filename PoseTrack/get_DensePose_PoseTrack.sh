#!/bin/bash
mkdir DensePose_PoseTrack
cd DensePose_PoseTrack
wget -O densepose_only_posetrack_train2017.json https://www.dropbox.com/s/tpbaemzvlojo2iz/densepose_only_posetrack_train2017.json?dl=1
wget -O densepose_only_posetrack_val2017.json https://www.dropbox.com/s/43h43s0t3hkuogr/densepose_only_posetrack_val2017.json?dl=1
wget -O densepose_posetrack_test2017.json https://www.dropbox.com/s/48tkd4pr8aa3rex/densepose_posetrack_test2017.json?dl=1
wget -O densepose_posetrack_train2017.json https://www.dropbox.com/s/awbdp3v0dz4jcau/densepose_posetrack_train2017.json?dl=1
wget -O densepose_posetrack_val2017.json https://www.dropbox.com/s/6tdmqx0h6a04vzz/densepose_posetrack_val2017.json?dl=1
