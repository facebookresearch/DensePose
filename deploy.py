from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace
import numpy as np
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)


##############################
###        Settings        ###
##############################
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: e.g. 0 1 2 3')
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


##############################
###         flask          ###
##############################
from flask import Flask
from flask_jsonrpc import JSONRPC

# Flask application
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Flask-JSONRPC
jsonrpc = JSONRPC(app, '/api', enable_web_browsable_api=False)


##############################
###          model         ###
##############################
# logger = logging.getLogger(__name__)
merge_cfg_from_file('./configs/DensePose_ResNet101_FPN_32x8d_s1x-e2e.yaml')
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
DOWNLOAD_CACHE = './detectron-download-cache/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pkl'
model = infer_engine.initialize_model_from_cfg(DOWNLOAD_CACHE)


##############################
###      detect_image      ###
##############################
def detect_image(im_path, thresh=0.9):
    im = cv2.imread(im_path)
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
            model, im[:,:,::-1], None, timers=timers)
        
    if isinstance(cls_boxes, list):
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im, np.zeros(im.shape), np.zeros([im.shape[0],im.shape[1]])
    # IUV
    IUV_fields = cls_bodys[1]
    # init 
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0],im.shape[1]])
    K = 26
    ## inds
    inds = np.argsort(-boxes[:,4])
    ## 
    for i, ind in enumerate(inds):
        entry = boxes[ind,:]
        if entry[4] > thresh:
            entry=entry[0:4].astype(int)
            ####
            output = IUV_fields[ind]
            ####
            All_Coords_Old = All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]
            All_Coords_Old[All_Coords_Old==0]=output.transpose([1,2,0])[All_Coords_Old==0]
            All_Coords[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2],:]= All_Coords_Old
            ###
            CurrentMask = (output[0,:,:]>0).astype(np.float32)
            All_inds_old = All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0]*(i+1)
            All_inds[ entry[1] : entry[1]+output.shape[1],entry[0]:entry[0]+output.shape[2]] = All_inds_old
    # Result processing
    All_Coords[:,:,1:3] = 255. * All_Coords[:,:,1:3]
    All_Coords[All_Coords>255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    All_inds = All_inds.astype(np.uint8)
    return im, All_Coords, All_inds

##############################
###          main          ###
##############################
# jsonrpc method
@jsonrpc.method('main')
def main(img_dir, output_dir, img_format='png', thresh=0.9):
    # get img paths
    img_paths = sorted(glob.glob(os.path.join(img_dir,'*'+img_format)))
    print('image nums:', len(img_paths))
    # detect
    for img_path in tqdm(img_paths):
        im, im_IUV, im_INDS = detect_image(img_path, thresh)
        idx = img_path.split('/')[-1].split('_')[-1].split('.')[0]
        output_path = os.path.join(output_dir, 'densepose'+'_'+idx+'.png')
        cv2.imwrite(output_path, im_IUV)

if __name__ == '__main__':
    app.run(host=args.host, port=args.port, debug=True)
