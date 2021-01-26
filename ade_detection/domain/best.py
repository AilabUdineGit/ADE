#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.domain.train_config import TrainConfig
from ade_detection.domain.split import Split
from ade_detection.domain.enums import *


class Best(object):
    

    def __init__(self, strict, partial):
        self.strict = strict
        self.partial = partial


    def __eq__(self, other):
        return self.partial == other.partial and \
               self.strict == other.strict 