# Results Format

This page describes the results format used by PoseTrack DensePose evaluation
procedure. The results format mimics the annotation format detailed on
the [data format](data_format.md) page. Please review the annotation
format before proceeding.

Each algorithmically generated result is stored separately in its own
result struct. This singleton result struct must contain the id of the
image from which the result was generated (a single image will typically
have multiple associated results). Results for the whole dataset are
aggregated in a single array. Finally, this entire result struct array
is stored to disk as a single JSON file (saved via
[gason](https://github.com/cocodataset/cocoapi/blob/master/MatlabAPI/gason.m)
in Matlab or [json.dump](https://docs.python.org/2/library/json.html) in Python).

Example result JSON files are available in
[example results](example_results.json).

The data struct for each of the result types is described below. The format
of the individual fields below (`category_id`, `bbox`, etc.) is the same as
for the annotation (for details see the [data format](data_format.md) page).
Bounding box coordinates `bbox` are floats measured from the top left image
corner (and are 0-indexed). We recommend rounding coordinates to the nearest
tenth of a pixel to reduce the resulting JSON file size. The dense estimates
of patch indices and coordinates in the UV space for the specified bounding
box are stored in `uv_shape` and `uv_data` fields.
`uv_shape` contains the shape of `uv_data` array, it should be of size
`(3, height, width)`, where `height` and `width` should match the bounding box
size. `uv_data` should contain PNG-compressed patch indices and U and V
coordinates scaled to the range `0-255`.

An example of code that generates results in the form of a `pkl` file can
be found in
[json_dataset_evaluator.py](https://github.com/facebookresearch/DensePose/blob/master/detectron/datasets/json_dataset_evaluator.py).
We also provide an [example script](../encode_results_for_competition.py) to convert
DensePose estimation results stored in a `pkl` file into a PNG-compressed
JSON file.




