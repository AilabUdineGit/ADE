#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
import pandas as pd
from os import path

import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm


class BaseImporter(object):


    def decompress_dataset(self, dataset_path, target_path):
        if not path.exists(target_path):
            LOG.info('dataset decompression in progress...')
            fm.decompress_zip(dataset_path)
            LOG.info('dataset decompressed successfully!')

            