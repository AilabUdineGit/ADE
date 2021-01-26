#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from transformers import AutoTokenizer
from transformers import AutoConfig
from os import path

from ade_detection.domain.train_config import TrainConfig
import ade_detection.utils.file_manager as fm
from ade_detection.domain.task import Task
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class ModelService(object):


    def __init__(self, task:Task):
        self.task = task


    def get_config(self):
        num_labels = (2 * len(self.task.goal)) + 1
        overrides = {
            'num_labels': num_labels,
            'output_attentions' : False,
            'output_hidden_states' : False,
            'hidden_dropout_prob' : self.task.train_config.dropout,
            'attention_probs_dropout_prob' :  self.task.train_config.dropout
        } # note: max_position_embeddings = 64 causes model issues
        config = AutoConfig.from_pretrained(self.task.model.value, **overrides)
        config.dropout = self.task.train_config.dropout
        config.num_labels = num_labels
        config.batch_size = BATCH_SIZE[self.task.corpus]
        config.model = self.task.model.value
        return config


    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.task.model.value, config=self.get_config())


    @staticmethod
    def get_bio_git_model():
        zip_path = loc.abs_path([loc.TMP, loc.BIO_BERT_ZIP])
        if not path.exists(zip_path):
            LOG.info('Model download in progress...')
            fm.wget_with_progressbar(loc.BIO_BERT_GIT_LINK, zip_path)
            LOG.info('Model download completed!')
        LOG.info('Model decompression in progress...')
        fm.decompress_zip(zip_path)
        LOG.info('Model decompression completed!')