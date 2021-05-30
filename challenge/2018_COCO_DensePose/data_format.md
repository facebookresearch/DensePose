# Data Format

The annotations are stored in [JSON](http://json.org/). Please note that
[COCO API](https://github.com/cocodataset/cocoapi) described on the
[download](http://cocodataset.org/#download) page can be used to access
and manipulate all annotations.

The annotations file structure is outlined below:
```
{
    "images" : [image],
    "annotations" : [annotation],
    "categories" : [category]
}

image {
    "coco_url" : str,
    "date_captured" : datetime,
    "file_name" : str,
    "flickr_url" : str,
    "id" : int,
    "width" : int,
    "height" : int,
    "license" : int
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
    "segmentation": RLE or [polygon]
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

Each dense pose annotation contains a series of fields, including the category
id and segmentation mask of the person. The segmentation format depends on
whether the instance represents a single object (`iscrowd=0` in which case
polygons are used) or a collection of objects (`iscrowd=1` in which case RLE
is used). Note that a single object (`iscrowd=0`) may require multiple polygons,
for example if occluded. Crowd annotations (`iscrowd=1`) are used to label large
groups of objects (e.g. a crowd of people). In addition, an enclosing bounding
box is provided for each person (box coordinates are measured from the top left
image corner and are 0-indexed).

The categories field of the annotation structure stores the mapping of category
id to category and supercategory names. It also has two fields: "keypoints",
which is a length `k` array of keypoint names, and "skeleton", which defines
connectivity via a list of keypoint edge pairs and is used for visualization.

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

