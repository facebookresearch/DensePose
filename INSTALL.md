# Installing DensePose

The DensePose-RCNN system is implemented within the [`detectron`](https://github.com/facebookresearch/Detectron) framework. This document is based on the Detectron installation instructions, for troubleshooting please refer to the [`detectron installation document`](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md).

**Requirements:**

- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages, and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- Detectron operators currently do not have CPU implementation; a GPU system is required.
- Detectron has been tested extensively with CUDA 8.0 and cuDNN 6.0.21.

## Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

## Other Dependencies

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

Note that instructions like `# COCOAPI=/path/to/install/cocoapi` indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (`COCOAPI` in this case) accordingly.

## Densepose

Clone the Densepose repository:

```
# DENSEPOSE=/path/to/clone/densepose
git clone https://github.com/facebookresearch/densepose $DENSEPOSE
```

Install Python dependencies:

```
pip install -r $DENSEPOSE/requirements.txt
```

Set up Python modules:

```
cd $DENSEPOSE && make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](tests/test_spatial_narrow_as_op.py)):

```
python2 $DENSEPOSE/detectron/tests/test_spatial_narrow_as_op.py
```

Build the custom operators library:

```
cd $DENSEPOSE && make ops
```

Check that the custom operator tests pass:

```
python2 $DENSEPOSE/detectron/tests/test_zero_even_op.py
```
### Fetch DensePose data.
Get necessary files to run, train and evaluate DensePose.
```
cd $DENSEPOSE/DensePoseData
bash get_densepose_uv.sh
```
For training, download the DensePose-COCO dataset:
```
bash get_DensePose_COCO.sh
```
For evaluation, get the necessary files:
```
bash get_eval_data.sh
```
## Setting-up the COCO dataset.

Create a symlink for the COCO dataset in your `datasets/data` folder.
```
ln -s /path/to/coco $DENSEPOSE/detectron/datasets/data/coco
```

Create symlinks for the DensePose-COCO annotations

```
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
ln -s $DENSEPOSE/DensePoseData/DensePose_COCO/densepose_coco_2014_valminusminival.json $DENSEPOSE/detectron/datasets/data/coco/annotations/
```

Your local COCO dataset copy at `/path/to/coco` should have the following directory structure:

```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

## Docker Image

We provide a [`Dockerfile`](docker/Dockerfile) that you can use to build a Densepose image on top of a Caffe2 image that satisfies the requirements outlined at the top. If you would like to use a Caffe2 image different from the one we use by default, please make sure that it includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).

Build the image:

```
cd $DENSEPOSE/docker
docker build -t densepose:c2-cuda9-cudnn7 .
```

Run the image (e.g. for [`BatchPermutationOp test`](tests/test_batch_permutation_op.py)):

```
nvidia-docker run --rm -it densepose:c2-cuda9-cudnn7 python2 detectron/tests/test_batch_permutation_op.py
```

To run inference in a docker container based on the prepared docker image,
one would need to make all the required data accessible within the container.
For that one should first follow the steps described in [Fetch DensePose Data](#fetch-densepose-data)
section. One could also prefetch all the necessary weights files used for training / inference.
Then one should start a container with the host `DensePoseData` and COCO directories mounted:

```
nvidia-docker run -v $DENSEPOSE/DensePoseData:/denseposedata -v /path/to/coco:/coco -it densepose:c2-cuda9-cudnn7 bash
```

Within the container one needs to replace the local `DensePoseData` directory with the host one:

```
mv /densepose/DensePoseData /densepose/DensePoseDataLocal
ln -s /denseposedata DensePoseData
```

and perform steps described in [COCO dataset setup](#setting-up-the-coco-dataset):

```
ln -s /coco /densepose/detectron/datasets/data/coco
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json /densepose/detectron/datasets/data/coco/annotations/
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json /densepose/detectron/datasets/data/coco/annotations/
ln -s /densepose/DensePoseData/DensePose_COCO/densepose_coco_2014_valminusminival.json /densepose/detectron/datasets/data/coco/annotations/
```

exit the container and commit the change

```
docker commit $(docker ps --last 1 -q) densepose:c2-cuda9-cudnn7-wdata
```

The new image can be used to run inference / training. However, one needs to
remember to mount `DensePoseData` and COCO directories:

```
nvidia-docker run --rm -v $DENSEPOSE/DensePoseData:/denseposedata -v /path/to/coco:/coco -it densepose:c2-cuda9-cudnn7-wdata <inference_or_training_command>
```

