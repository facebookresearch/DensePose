# Data Format

The annotations are stored in [JSON](http://json.org/). Please note that
[COCO API](https://github.com/facebookresearch/DensePose/blob/master/detectron/datasets/densepose_cocoeval.py)
can be used to access and manipulate all annotations.

The annotations file structure is outlined below:
```
{
    "images" : [image],
    "annotations" : [annotation],
    "categories" : [category]
}

image {
    "id" : int,
    "width" : int,
    "height" : int,
    "file_name" : str,
    "has_no_densepose": True or False,
    "is_labeled": True or False,
    "vid_id": int,
    "frame_id": int,
    "nframes": int,
    "ignore_regions_x": [[int]],
    "ignore_regions_y": [[int]]
}

annotation {
    "area": float,
    "bbox": [x, y, width, height],
    "category_id": int,
    "dp_I": [float],
    "dp_U": [float],
    "dp_V": [float],
    "dp_masks": [dp_mask],
    "dp_x": [float],
    "dp_y": [float],
    "id": int,
    "image_id": int,
    "iscrowd": 0 or 1,
    "keypoints": [float],
    "segmentation": RLE or [polygon],
    "track_id": int
}

category {
    "id" : int,
    "name" : str,
    "supercategory" : str,
    "keypoints": [str],
    "skeleton": [edge]
}

dp_mask {
    "counts": str,
    "size": [int, int]
}
```

Annotation data consists of 3 fields: `images`, `annotations` and `categories`.

The `images` field contains a list of image data entries with image ID (`id`),
image size (`width`, `height`), image file name (`file_name`), video ID
(`vid_id`), image frame ID in the video (`frame_id`) and the total number 
of frames in the video sequence (`nframes`).
The flag `is_labeled` indicates indicate whether the image was annotated,
`has_no_densepose` is set to `True` if the corresponding DensePose data should
not be considered in the evaluation. If an image is meant to be used in
the evaluation, it either has no entry for `has_no_densepose`, or has it set to
`False`. For more fine-grained control over evaluation, image regions
that should be excluded from the evaluation might be specified through
`ignore_regions_x` and `ignore_regions_y`.
Each region is given as two arrays of integer `x` and `y` coordinates, which
correspond to polygon vertex coordinates.

The `annotations` field contains a list of annotations.
Each DensePose annotation contains a series of fields, including the category
id and segmentation mask of the person. The segmentation format depends on
whether the instance represents a single object (`iscrowd=0` in which case
polygons are used) or a collection of objects (`iscrowd=1` in which case RLE
is used). Note that a single object (`iscrowd=0`) may require multiple polygons,
for example if occluded. Crowd annotations (`iscrowd=1`) are used to label large
groups of objects (e.g. a crowd of people). In addition, an enclosing bounding
box and `track_id` are provided for each person (box coordinates are measured from
the top left image corner and are 0-indexed).

DensePose annotations are stored in `dp_*` fields:

*Annotated masks*:

* `dp_masks`: RLE encoded dense masks. All part masks are of size 256x256.
They correspond to 14 semantically meaningful parts of the body: `Torso`,
`Right Hand`, `Left Hand`, `Left Foot`, `Right Foot`, `Upper Leg Right`,
`Upper Leg Left`, `Lower Leg Right`, `Lower Leg Left`, `Upper Arm Left`,
`Upper Arm Right`, `Lower Arm Left`, `Lower Arm Right`, `Head`;

*Annotated points*:

* `dp_x`, `dp_y`: spatial coordinates of collected points on the image.
The coordinates are scaled such that the bounding box size is 256x256;
* `dp_I`: The patch index that indicates which of the 24 surface patches the
point is on. Patches correspond to the body parts described above. Some
body parts are split into 2 patches: `1, 2 = Torso`, `3 = Right Hand`,
`4 = Left Hand`, `5 = Left Foot`, `6 = Right Foot`, `7, 9 = Upper Leg Right`,
`8, 10 = Upper Leg Left`, `11, 13 = Lower Leg Right`, `12, 14 = Lower Leg Left`,
`15, 17 = Upper Arm Left`, `16, 18 = Upper Arm Right`, `19, 21 = Lower Arm Left`,
`20, 22 = Lower Arm Right`, `23, 24 = Head`;
* `dp_U`, `dp_V`: Coordinates in the UV space. Each surface patch has a
separate 2D parameterization.

The categories field of the annotation structure stores the mapping of category
id to category and supercategory names. It also has two fields: "keypoints",
which is a length `k` array of keypoint names, and "skeleton", which defines
connectivity via a list of keypoint edge pairs and is used for visualization.


