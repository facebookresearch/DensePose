# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""IO utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import hashlib
import logging
import os
import re
import sys
import urllib2

logger = logging.getLogger(__name__)


def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def cache_url(url_or_file, cache_dir):
    """Download the file specified by the URL to the cache_dir and return the
    path to the cached file. If the argument is not a URL, simply return it as
    is.
    """
    if re.match(r'^\$', url_or_file, re.IGNORECASE) is not None:
        url_or_file = os.path.expandvars(url_or_file)
        assert os.path.exists(url_or_file)
        return url_or_file
    elif re.match(r'^~', url_or_file, re.IGNORECASE) is not None:
        url_or_file = os.path.expanduser(url_or_file)
        assert os.path.exists(url_or_file)
        return url_or_file

    is_url = re.match(r'^(?:http)s?://', url_or_file, re.IGNORECASE) is not None

    if not is_url:
        return url_or_file
    #
    url = url_or_file
    #
    Len_filename = len(url.split('/')[-1])
    BASE_URL = url[0:-Len_filename - 1]
    #
    cache_file_path = url.replace(BASE_URL, cache_dir)
    if os.path.exists(cache_file_path):
        #assert_cache_file_is_ok(url, cache_file_path)
        return cache_file_path

    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)

    logger.info('Downloading remote file {} to {}'.format(url, cache_file_path))
    download_url(url, cache_file_path)
    #assert_cache_file_is_ok(url, cache_file_path)
    return cache_file_path


def assert_cache_file_is_ok(url, file_path):
    """Check that cache file has the correct hash."""
    # File is already in the cache, verify that the md5sum matches and
    # return local path
    cache_file_md5sum = _get_file_md5sum(file_path)
    ref_md5sum = _get_reference_md5sum(url)
    assert cache_file_md5sum == ref_md5sum, \
        ('Target URL {} appears to be downloaded to the local cache file '
         '{}, but the md5 hash of the local file does not match the '
         'reference (actual: {} vs. expected: {}). You may wish to delete '
         'the cached file and try again to trigger automatic '
         'download.').format(url, file_path, cache_file_md5sum, ref_md5sum)


def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(
        '  [{}] {}% of {:.1f}MB file  \r'.
        format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write('\n')


def download_url(
    url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar
):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    response = urllib2.urlopen(url)
    total_size = response.info().getheader('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, 'wb') as f:
        while 1:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)

    return bytes_so_far


def _get_file_md5sum(file_name):
    """Compute the md5 hash of a file."""
    hash_obj = hashlib.md5()
    with open(file_name, 'r') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


def _get_reference_md5sum(url):
    """By convention the md5 hash for url is stored in url + '.md5sum'."""
    url_md5sum = url + '.md5sum'
    md5sum = urllib2.urlopen(url_md5sum).read().strip()
    return md5sum
