#!/usr/bin/env python2

"""encode_results.py: script to encode dense human pose estimation results
in DensePose format into a packed representation using PNG compression.
"""

__author__    = "Vasil Khalidov"
__copyright__ = "Copyright (c) 2018-present, Facebook, Inc."

import os
import sys
import pickle
import copy
import json
import time
import argparse
import numpy as np

kPositiveAnswers = ['y', 'Y']
kNegativeAnswers = ['n', 'N']
kAnswers = kPositiveAnswers + kNegativeAnswers

def _parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('inPklResultsFile', help='Input pickle file with'
        ' dense human pose estimation results')
    parser.add_argument('outJsonPackedFile', help='Output JSON file with'
        ' packed dense human pose estimation results, which can be'
        ' used for submission')
    args = parser.parse_args()
    return args

def _encodePngData(arr):
    """
    Encode array data as a PNG image using the highest compression rate
    @param arr [in] Data stored in an array of size (3, M, N) of type uint8
    @return Base64-encoded string containing PNG-compressed data
    """
    from PIL import Image
    import StringIO
    assert len(arr.shape) == 3, "Expected a 3D array as an input," \
            " got a {0}D array".format(len(arr.shape))
    assert arr.shape[0] == 3, "Expected first array dimension of size 3," \
            " got {0}".format(arr.shape[0])
    assert arr.dtype == np.uint8, "Expected an array of type np.uint8, " \
            " got {0}".format(arr.dtype)
    data = np.moveaxis(arr, 0, -1)
    im = Image.fromarray(data)
    fStream = StringIO.StringIO()
    im.save(fStream, format='png', optimize=True)
    s = fStream.getvalue()
    return s.encode('base64')

def _statusStr(i, dataLen):
    kProgressWidth = 20
    kProgressTemplate = '[{0}] {1: 4d}%'
    progressVisNDone = min(max(0, i * kProgressWidth // dataLen),
        kProgressWidth)
    progressVisNTodo = kProgressWidth - progressVisNDone 
    progressVis = '*' * progressVisNDone + ' ' * progressVisNTodo
    progressNum = i * 100 // dataLen
    progressStr = kProgressTemplate.format(progressVis, progressNum)
    return progressStr

def _savePngJson(hInPklResultsFile, hOutJsonPackedFile):
    from PIL import Image
    import StringIO
    dataFPickle = pickle.load(hInPklResultsFile)
    statusStr = ''
    dataLen = len(dataFPickle)
    for i, x in enumerate(dataFPickle):
        x['uv_shape'] = x['uv'].shape
        x['uv_data'] = _encodePngData(x['uv'])
        del x['uv']
        sys.stdout.write('\b' * len(statusStr))
        statusStr = _statusStr(i, dataLen)
        sys.stdout.write(statusStr)
    sys.stdout.write('\n')
    json.dump(dataFPickle, hOutJsonPackedFile, ensure_ascii=False,
        sort_keys=True, indent=4)

def main():
    args = _parseArguments()
    if os.path.exists(args.outJsonPackedFile):
        answer = ''
        while not answer in kAnswers:
            answer = raw_input('File "{0}" already exists, overwrite? [y/n] '
                .format(args.outJsonPackedFile))
        if answer in kNegativeAnswers:
            sys.exit(1)

    with open(args.inPklResultsFile, 'rb') as hIn, \
            open(args.outJsonPackedFile, 'w') as hOut:
        print('Encoding png: {0}'.format(args.outJsonPackedFile))
        start = time.clock()
        _savePngJson(hIn, hOut)
        end = time.clock()
        print('Finished encoding png, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()
