#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW
from tqdm import tqdm
from os import path
import pandas as pd
import numpy as np
import random
import pickle
from torch import LongTensor
import torch
import gc
import os


import ade_detection.utils.localizations as loc
from ade_detection.domain.enums import *
import ade_detection.utils.file_manager as fm
from ade_detection.domain.task import Task
from ade_detection.domain.train_config import TrainConfig
from ade_detection.domain.span import Span
from ade_detection.domain.subtoken import SubToken
from ade_detection.domain.document import Document
from ade_detection.services.database_service import DatabaseService
from ade_detection.services.split_service import SplitService
from ade_detection.services.query_service import QueryService
from ade_detection.services.model_service import ModelService
from ade_detection.models.base_task_loader import BaseTaskLoader


class BertTaskLoader(BaseTaskLoader):


    def __init__(self, task: Task):
        super(BertTaskLoader, self).__init__(task)

        split_svc = SplitService()
        self.task.split = split_svc.load_split(self.task)
        
        model_svc = ModelService(task)
        self.tokenizer = model_svc.get_tokenizer()

        self.task.split.train = self.load(task.split.train)
        self.task.split.test = self.load(task.split.test)
        self.task.split.validation = self.load(task.split.validation)


    def load(self, sdocs):
        for annotation_type in self.task.goal:
            if TIDY_MODE.MERGE_OVERLAPS in self.task.tidy_modes and \
               TIDY_MODE.SOLVE_DISCONTINUOUS in self.task.tidy_modes:
                sdocs = self.merge_discontinuous_overlaps(sdocs, annotation_type)
                sdocs = self.solve_discontinuous(sdocs, annotation_type)
                sdocs = self.merge_overlaps(sdocs, annotation_type)
            else:
                raise NotImplementedError('tidy mode combination not implemented') 
        i = 0
        while i < len(sdocs):
            sdoc = sdocs[i]
            if sdoc.subtokens == None:
                sdoc = self.subtokenize(sdoc, self.task, self.tokenizer)
                sdoc = self.subtokens_biluo_tagging(sdoc, self.task)
                if self.task.notation == NOTATION.IOB:
                    sdoc.tags = self.biluo_to_iob(sdoc.tags)
                if self.task.notation == NOTATION.IO:
                    sdoc.tags = self.biluo_to_io(sdoc.tags)       
            if len(sdoc.subtokens) > self.max_seq_len - 2: 
                new_sdoc = sdoc.copy()
                new_sdoc.id = sdocs[-1].id + 1 
                (char_index, subtoken_index) = self.find_split_index(sdoc) 
                sdoc.subtokens = sdoc.subtokens[:subtoken_index]
                new_sdoc.subtokens = new_sdoc.subtokens[subtoken_index:]
                sdoc.tags = sdoc.tags[:subtoken_index]
                new_sdoc.tags = new_sdoc.tags[subtoken_index:]
                sdoc.text = sdoc.text[:char_index]
                new_sdoc.text = new_sdoc.text[char_index:]
                sdocs.append(new_sdoc)
            i += 1
        sdocs = self.numericalize(sdocs, self.tokenizer)
        return sdocs
