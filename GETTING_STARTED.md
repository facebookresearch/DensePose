# Using Detectron

This document provides brief tutorials covering DensePose for inference and training on the DensePose-COCO dataset.
This document is a modified version of the [`detectron/GETTING_STARTED.md`](https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md).

- For general information about DensePose, please see [`README.md`](README.md).
- For installation instructions, please see [`INSTALL.md`](INSTALL.md).

## Inference with Pretrained Models

#### 1. Directory of Image Files
To run inference on a an image (or a directory of image files), you can use the `infer_simple.py` tool. In this example, we're using an end-to-end trained DensePose-RCNN model with a ResNet-101-FPN backbone from the model zoo:
```
python2 tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/demo_im.jpg
```

DensePose should automatically download the model from the URL specified by the `--wts` argument. This tool will output visualizations of the detections in PDF format in the directory specified by `--output-dir`. Also, it will output two images `*_IUV.png` and `*_INDS.png` which consists of I,U, V channels and segmented instance indices respectively. Please see [`notebooks/DensePose-RCNN-Visualize-Results.ipynb`](notebooks/DensePose-RCNN-Visualize-Results.ipynb) for the visualizations of these outputs.


## Testing with Pretrained Models

Make sure that you have downloaded the DensePose evaluation files as instructed in [`INSTALL.md`](INSTALL.md). 
This example shows how to run an end-to-end trained DensePose-RCNN model from the model zoo using a single GPU for inference. As configured, this will run inference on all images in `coco_2014_minival` (which must be properly installed).

```
python2 tools/test_net.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    NUM_GPUS 1
```

## Training a Model

This example shows how to train a model using the DensePose-COCO dataset. The model will be an end-to-end trained DensePose-RCNN using a ResNet-50-FPN backbone. 

```
python2 tools/train_net.py \
    --cfg configs/DensePose_ResNet50_FPN_single_GPU.yaml \
    OUTPUT_DIR /tmp/detectron-output
```
The models we have provided in the model zoo are trained using 8 gpus. As in any  Detectron configs, we use linear scaling rule to adjust learning schedules. Please refer to the following paper: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). We also provide learning rate and number of iterations for varying number of GPUs.

