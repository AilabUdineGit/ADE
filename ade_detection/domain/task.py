#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.domain.train_config import TrainConfig
from ade_detection.domain.split import Split
from ade_detection.domain.enums import *


class Task(object):
    

    def __init__(self, id: str, split_folder: str,tidy_modes: list, corpus: CORPUS, 
                 notation: NOTATION, model: MODEL, architecture: ARCHITECTURE, 
                 goal: list, train_mode: TRAIN_MODE, train_config: TrainConfig,  
                 split: Split = None):
        self.id = id
        self.split_folder = split_folder
        self.split = split
        self.tidy_modes = tidy_modes
        self.corpus = corpus
        self.notation = notation
        self.model = model
        self.architecture = architecture
        self.train_config = train_config
        self.goal = goal
        self.train_mode = train_mode
           

    def __eq__(self, other):
        return self.id == other.id and \
               self.split_folder == other.split_folder and \
               self.split == other.split and \
               self.tidy_modes == other.tidy_modes and \
               self.notation == other.notation and \
               self.model == other.model and \
               self.architecture == other.architecture and \
               self.train_config == other.train_config and \
               self.goal == other.goal and \
               self.train_mode == other.train_mode 