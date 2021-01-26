#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from torch.utils.data import TensorDataset
from torch import LongTensor
import torch

from ade_detection.domain.enums import *


class Split(object):


    def __init__(self, train: list, test: list, validation: list,
                 train_tensor = None, test_tensor = None, validation_tensor = None):
        self.train = train
        self.test = test
        self.validation = validation
        
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.validation_tensor = validation_tensor
           

    def to_tensor_dataset(self, sdocs):
        return TensorDataset( LongTensor([x.num_subtokens for x in sdocs]),
                              LongTensor([x.attention_mask for x in sdocs]),
                              LongTensor([x.num_tags for x in sdocs]),
                              LongTensor([x.id for x in sdocs]) )


    def __eq__(self, other):
        return self.train == other.train and \
               self.test == other.test and \
               self.validation == other.validation