# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This is a modified version of cocoeval.py where we also have the densepose evaluation.

__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy
import h5py
import pickle
from scipy.io import loadmat
import scipy.spatial.distance as ssd
import os
import itertools

class denseposeCOCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox', 'keypoints' or 'uv'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', sigma=1.):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        if iouType == 'uv':
            self.sigma = sigma
        self.ignoreThrBB = 0.7
        self.ignoreThrUV = 0.9
        self.num_parts = 24  # number of pre-defined body parts

    def _loadGEval(self):
        prefix = os.path.dirname(__file__) + '/../../DensePoseData/eval_data/'
        print('Loading densereg GT from {}'.format(prefix))
        SMPL_subdiv = loadmat(prefix + 'SMPL_subdiv.mat')
        self.PDIST_transform = loadmat(prefix + 'SMPL_SUBDIV_TRANSFORM.mat')
        # 1-based index of geodesic distance matrix, 
        # range: [1, 27554], shape: (num_total_points=29408,)
        self.PDIST_transform = self.PDIST_transform['index'].squeeze()
        # UV coordinates of all collected points on SMPL model, shape: (2, num_total_points)
        UV = np.array([SMPL_subdiv['U_subdiv'], SMPL_subdiv['V_subdiv']]).squeeze()
        # body part index (1 ~ 24), shape: (num_total_point,)
        Part_ids = SMPL_subdiv['Part_ID_subdiv'].squeeze()
        self.Part_ids = np.array(Part_ids)
        # 1-based index of closest vertex index of each point,
        # range: [1, num_total_points], shape: (num_total_points,)
        ClosestVertInds = np.arange(UV.shape[1]) + 1
        self.Part_UVs = []
        self.Part_ClosestVertInds = []
        # UV coordinates and closest vertex indices of points in each part
        for i in np.arange(1, self.num_parts + 1):
            self.Part_UVs.append(UV[:, Part_ids == i])
            self.Part_ClosestVertInds.append(ClosestVertInds[Part_ids == i])

        arrays = {}
        f = h5py.File(prefix + 'Pdist_matrix.mat')
        for k, v in f.items():
            arrays[k] = np.array(v)
        f.close()
        # precomputed geodesic distances matrix with compact representation
        self.Pdist_matrix = arrays['Pdist_matrix']
        # Mean geodesic distances for parts.
        self.Mean_Distances = np.array([0, 0.351, 0.107, 0.126, 0.237, 0.173, 0.142, 0.128, 0.150])
        # Coarse Part labels.
        self.CoarseParts = np.array(
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]
        )

        print('densereg GT loaded')

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        def _getIgnoreRegion(iid, coco):
            img = coco.imgs[iid]

            if not 'ignore_regions_x' in img.keys():
                return None

            if len(img['ignore_regions_x']) == 0:
                return None

            rgns_merged = []
            for region_x, region_y in zip(img['ignore_regions_x'], img['ignore_regions_y']):
                rgns = [iter(region_x), iter(region_y)]
                rgns_merged.append(list(it.next() for it in itertools.cycle(rgns)))
            rles = maskUtils.frPyObjects(rgns_merged, img['height'], img['width'])
            rle = maskUtils.merge(rles)
            return maskUtils.decode(rle)

        def _checkIgnore(dt, iregion):
            if iregion is None:
                return True

            bb = np.array(dt['bbox']).astype(np.int)
            x1, y1, x2, y2 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            x2 = min([x2, iregion.shape[1]])
            y2 = min([y2, iregion.shape[0]])

            if bb[2] * bb[3] == 0:
                return False

            crop_iregion = iregion[y1:y2, x1:x2]

            if crop_iregion.sum() == 0:
                return True

            if not 'uv' in dt.keys(): # filtering boxes
                return crop_iregion.sum() / bb[2] / bb[3] < self.ignoreThrBB

            # filtering UVs
            ignoremask = np.require(crop_iregion, requirements=['F'])
            uvmask = np.require(np.asarray(dt['uv'][0] > 0), dtype=np.uint8,
                    requirements=['F'])
            uvmask_ = maskUtils.encode(uvmask)
            ignoremask_ = maskUtils.encode(ignoremask)
            uviou = maskUtils.iou([uvmask_], [ignoremask_], [1])[0]
            return uviou < self.ignoreThrUV

        p = self.params

        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # add point gt annotations if iouType == 'uv'
        if p.iouType == 'uv':
            self._loadGEval()

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
            if p.iouType == 'uv':
                gt['ignore'] = 'dp_x' not in gt

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self._igrgns = defaultdict(list)

        for gt in gts:
            iid = gt['image_id']
            if not iid in self._igrgns.keys():
                self._igrgns[iid] = _getIgnoreRegion(iid, self.cocoGt)
            if _checkIgnore(gt, self._igrgns[iid]):
                self._gts[iid, gt['category_id']].append(gt)
        for dt in dts:
            if _checkIgnore(dt, self._igrgns[dt['image_id']]):
                self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}                     # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType in ['segm', 'bbox']:
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        elif p.iouType == 'uv':
            computeIoU = self.computeGps

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId)
                for imgId in p.imgIds
                for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet) 
                for catId in catIds
                for areaRng in p.areaRng
                for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be N x m
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def computeGps(self, imgId, catId):
        p = self.params
        # dimention here should be N x m
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if len(gt) == 0 or len(dt) == 0:
            return []
        ious = np.zeros((len(dt), len(gt)))

        # compute gps between each detection and ground truth object
        # sigma = self.sigma  # 0.255 # dist = 0.3m corresponds to gps = 0.5
        # 1 # dist = 0.3m corresponds to gps = 0.96
        # 1.45 # dist = 1.7m (person height) corresponds to gps = 0.5)
        for j, g in enumerate(gt):
            # gps between any detection and an ignored ground truth person is 0
            if g['ignore']:
                continue
            bb = g['bbox']
            for i, d in enumerate(dt):
                # width and height of detected bbox
                dx = d['bbox'][2]; dy = d['bbox'][3]
                # (2D) spatial coordinates of annotated points within current gt bbox on the image, 
                # which are scaled such that the gt bbox size is 256 x 256.
                dp_x = np.array(g['dp_x']) * bb[2] / 255.
                dp_y = np.array(g['dp_y']) * bb[3] / 255.
                # spatial coordinates of annotated points relative to detected bbox
                px = (dp_x + bb[0] - d['bbox'][0]).astype(np.int)
                py = (dp_y + bb[1] - d['bbox'][1]).astype(np.int)
                pts = np.zeros(len(dp_x))  # len(dp_x): number of annotated points
                # annotated points outside the range of detected bbox are considered as background
                pts[px >= dx] = -1; pts[py >= dy] = -1
                pts[px < 0] = -1; pts[py < 0] = -1
                # print("#collected gt points: ", len(dp_x))
                if len(dp_x) == 0:
                    gps = 0.
                elif pts.max() == -1:
                    gps = 0.
                else:
                    px[pts == -1] = 0; py[pts == -1] = 0;
                    ipoints = d['uv'][0, py, px]
                    upoints = d['uv'][1, py, px] / 255. # convert from uint8 by /255.
                    vpoints = d['uv'][2, py, px] / 255.
                    ipoints[pts == -1] = 0
                    # Find closest vertices index in subsampled mesh.
                    cVertInds, cVertIndsGT = self.findAllClosestVertInds(g, upoints, vpoints, ipoints)
                    # Get pairwise geodesic distances between GT and estimated mesh points.
                    dist = self.getDistances(cVertInds, cVertIndsGT)
                    # Compute the GPS measure.
                    if len(dist) > 0:
                        # Find the mean geodesic normalization distance for each GT point, based on which part it is on.
                        Current_Mean_Distances  = self.Mean_Distances[
                            self.CoarseParts[self.Part_ids[cVertIndsGT[cVertIndsGT > 0] - 1]]
                        ]
                        # Compute gps
                        gps = np.exp(-(dist ** 2) / (2 * (Current_Mean_Distances ** 2)))
                        gps = np.sum(gps) / len(dist)
                    else:
                        gps = 0.
                ious[i, j] = gps

        gbb = [g['bbox'] for g in gt]
        dbb = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious_bb = maskUtils.iou(dbb, gbb, iscrowd)
        return ious, ious_bb

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        if p.iouType == 'uv':
            ious = self.ious[imgId, catId][0][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ioubs = self.ious[imgId, catId][1][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        else:
            ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T, G))
        dtm  = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if np.all(gtIg) == True and p.iouType == 'uv':
            dtIg = np.logical_or(dtIg, True)

        if len(ious) > 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        ## redundant condition after the above one
                        # if ious[dind, gind] == 0.:
                        #     continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind]  = gt[m]['id']
                    gtm[tind, m]     = d['id']

        """
        When evaluating for body_uv, suppressing/ignoring a detected box (`dbox`) at all GPS thresholds 
        which satisfies the following criterion:
            GPS(dbox, gbox) = 0 while IoU(dbox, gbox) > 0.8, where `gbox` is an ignored gt box.
        """
        if p.iouType == 'uv' and len(ioubs) > 0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                if dtm[tind, dind] == 0:
                    ioub = 0.8  # a manually set IoU threshold
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # continue to next gt unless better match made
                        if ioubs[dind, gind] < ioub:
                            continue
                        # if match successful and best so far, store appropriately
                        ioub = ioubs[dind, gind]
                        m = gind
                        # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[:, dind] = gtIg[m]
                    if gtIg[m]:
                        dtm[tind, dind] = gt[m]['id']
                        gtm[tind, m]    = d['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id':    imgId,
            'category_id': catId,
            'aRng':        aRng,
            'maxDet':      maxDet,
            'dtIds':       [d['id'] for d in dt],
            'gtIds':       [g['id'] for g in gt],
            'dtMatches':   dtm,
            'gtMatches':   gtm,
            'dtScores':    [d['score'] for d in dt],
            'gtIgnore':    gtIg,
            'dtIgnore':    dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T         = len(p.iouThrs)
        R         = len(p.recThrs)
        K         = len(p.catIds) if p.useCats else 1
        A         = len(p.areaRng)
        M         = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall    = -np.ones((T, K, A, M))
        scores    = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        print('Categories ids:', p.catIds)
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet]  for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        print('Final precisions, max: {:.2f}, min: {:.2f}'.format(np.max(precision), np.min(precision)))
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ {}={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            measure = 'IoU'
            if self.params.iouType == 'keypoints':
                measure = 'OKS'
            elif self.params.iouType == 'uv':
                measure = 'GPS'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(np.abs(iouThr - p.iouThrs) < 0.001)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, measure, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        def _summarizeUvs():
            stats = np.zeros((18,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.5)
            stats[2] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.55)
            stats[3] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.60)
            stats[4] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.65)
            stats[5] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.70)
            stats[6] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.75)
            stats[7] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.80)
            stats[8] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.85)
            stats[9] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.90)
            stats[10] = _summarize(1, maxDets=self.params.maxDets[0], iouThr=.95)
            stats[11] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='medium')
            stats[12] = _summarize(1, maxDets=self.params.maxDets[0], areaRng='large')
            stats[13] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[14] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.5)
            stats[15] = _summarize(0, maxDets=self.params.maxDets[0], iouThr=.75)
            stats[16] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='medium')
            stats[17] = _summarize(0, maxDets=self.params.maxDets[0], areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType in ['segm','bbox']:
            summarize = _summarizeDets
        elif iouType in ['keypoints']:
            summarize = _summarizeKps
        elif iouType in ['uv']:
            summarize = _summarizeUvs
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    # ================ functions for dense pose ==============================
    def findAllClosestVertInds(self, gt, U_points, V_points, Index_points):
        I_gt = np.array(gt['dp_I'])
        U_gt = np.array(gt['dp_U'])
        V_gt = np.array(gt['dp_V'])
        # find closest vertex for each estimated point and gt point in each part
        ClosestVertInds = np.ones(Index_points.shape, dtype=int) * -1
        ClosestVertIndsGT = np.ones(Index_points.shape, dtype=int) * -1
        for i in np.arange(1, self.num_parts + 1):
            Current_Part_UVs = self.Part_UVs[i - 1]
            Current_Part_ClosestVertInds = self.Part_ClosestVertInds[i - 1]
            if sum(Index_points == i) > 0:
                UVs = np.array([U_points[Index_points == i], V_points[Index_points == i]])
                D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
                ClosestVertInds[Index_points == i] = Current_Part_ClosestVertInds[np.argmin(D, axis=0)]
            if sum(I_gt == i) > 0:
                UVs = np.array([U_gt[I_gt == i], V_gt[I_gt == i]])
                D = ssd.cdist(Current_Part_UVs.transpose(), UVs.transpose()).squeeze()
                ClosestVertIndsGT[I_gt == i] = Current_Part_ClosestVertInds[np.argmin(D, axis=0)]

        return ClosestVertInds, ClosestVertIndsGT


    def getDistances(self, cVertInds, cVertIndsGT):
        cVerts = self.PDIST_transform[cVertInds - 1]
        cVertsGT = self.PDIST_transform[cVertIndsGT - 1]
        cVerts[cVertInds < 0] = 0
        cVertsGT[cVertIndsGT < 0] = 0
        # `n` is the number of points of which the geodesic distances are precomputed
        # n = 27554
        dists = []
        # loop through each pair of (gt_point, dt_point)
        for d in range(len(cVertsGT)):
            if cVertsGT[d] > 0:
                if cVerts[d] > 0:
                    i = cVertsGT[d] - 1
                    j = cVerts[d] - 1
                    if i == j:  # elements on the diagonal are all zeros
                        dists.append(0)
                    elif i > j:
                        # find the offset to fetch the precomputed geodesic distance
                        k = i * (i - 1) / 2 + j
                        dists.append(self.Pdist_matrix[k][0])
                    else:
                        k = j * (j - 1) / 2 + i
                        dists.append(self.Pdist_matrix[k][0])
                else:
                    dists.append(np.inf)
        return np.array(dists).squeeze()


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def setUvParams(self):
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        elif iouType == 'uv':
            self.setUvParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
