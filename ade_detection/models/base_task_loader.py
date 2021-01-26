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


from ade_detection.domain.train_config import TrainConfig
from ade_detection.services.database_service import DatabaseService
from ade_detection.services.split_service import SplitService
from ade_detection.services.query_service import QueryService
from ade_detection.services.model_service import ModelService
from ade_detection.domain.document import Document
from ade_detection.domain.subtoken import SubToken
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.span import Span
from ade_detection.domain.task import Task
from ade_detection.domain.enums import *


class BaseTaskLoader(object):


    def __init__(self, task: Task):
        self.task = task
        self.max_seq_len = MAX_SEQ_LEN[self.task.corpus]
        

    def numericalize(self, sdocs, tokenizer):
        for sdoc in sdocs:
            CLS = tokenizer.cls_token
            PAD = tokenizer.pad_token
            SEP = tokenizer.sep_token

            sdoc.subtokens.insert(0, SubToken(None, CLS))
            sdoc.tags.insert(0, 'O')
            sdoc.subtokens.append(SubToken(None, SEP))
            sdoc.tags.append('O')
            
            sdoc.subtokens.extend([SubToken(None, PAD)] * (self.max_seq_len - len(sdoc.subtokens)))
            sdoc.tags.extend(['O'] * (self.max_seq_len - len(sdoc.tags)))
            sdoc.attention_mask = [0 if x.text == PAD else 1 for x in sdoc.subtokens]
            sdoc.num_subtokens = tokenizer.convert_tokens_to_ids([x.text for x in sdoc.subtokens])
            sdoc.num_tags = self.convert_iob_tags_to_ids(sdoc.tags, sdoc.subtokens, tokenizer) 
            assert len(sdoc.subtokens) == len(sdoc.tags)
            assert len(sdoc.attention_mask) == len(sdoc.tags)
            assert len(sdoc.num_tags) == len(sdoc.tags)
            assert len(sdoc.num_subtokens) == len(sdoc.subtokens)
        return sdocs


    def convert_iob_tags_to_ids(self, tags, subtokens, tokenizer):
        num_tags = []
        for i, t in enumerate(tags):
            if subtokens[i].text == tokenizer.pad_token:
                num_tags.append(-1)
            else:
                if t == 'O':
                    num_tags.append(0)
                else:
                    annotation = t[2:]
                    index = self.index_by_annotation(annotation)
                    if t[0] == 'B':
                        num_tags.append(index * 2 + 2)
                    elif t[0] == 'I':
                        num_tags.append(index * 2 + 1)
        return num_tags


    def index_by_annotation(self, annotation):
        for i, a in enumerate(self.task.goal):
            if a.name == annotation:
                return i 
        assert False


    def find_split_index(self, sdoc):
        max_seq_len = self.max_seq_len - 2
        split_candidate = sdoc.subtokens[max_seq_len]
        ends = list(filter(lambda x: x <= split_candidate.token.end, [x.end for x in sdoc.doc.sentences]))
        end = max(ends) if len(ends) > 0 else split_candidate.token.end 
        subtoken_index = self.nearest_subtoken(sdoc.subtokens, end)
        return (sdoc.subtokens[subtoken_index].token.end, subtoken_index)


    def nearest_subtoken(self, array, value):
            winner = 0
            best_delta = abs(array[0].token.end - value)
            for i, s in enumerate(array):
                delta = abs(s.token.end - value)
                if delta <= best_delta:
                    best_delta = delta
                    winner = i 
            return winner


    def subtokens_biluo_tagging(self, sdoc, task):
        tags = ['O'] * len(sdoc.subtokens) 
        for span in sdoc.doc.spans:
            for annotation_type in task.goal:
                if span.contains_annotation(annotation_type):
                    begin = span.tokens[0].subtokens_interval[0]
                    end = span.tokens[-1].subtokens_interval[1]
                    if begin == end - 1:
                        tags[begin] = 'U-' + annotation_type.name
                    else:
                        tags[begin] = 'B-' + annotation_type.name 
                        tags[end - 1] = 'L-' + annotation_type.name 
                        for i in range(begin + 1, end - 1):
                            tags[i] = 'I-' + annotation_type.name
        sdoc.tags = tags
        return sdoc


    def tokens_single_biluo_tagging(self, sdoc, task):
        tags = ['O'] * len(sdoc.doc.tokens) 
        for span in sdoc.doc.spans:
            for annotation_type in task.goal:
                if span.contains_annotation(annotation_type):
                    begin = sdoc.doc.tokens.index(span.tokens[0])
                    end = sdoc.doc.tokens.index(span.tokens[-1])
                    if begin == end:
                        tags[begin] = 'U'
                    else:
                        tags[begin] = 'B'
                        tags[end] = 'L'
                        for i in range(begin + 1, end):
                            tags[i] = 'I'
        sdoc.tags = tags
        return sdoc


    def biluo_to_iob(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'B', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
        return biluo 


    def biluo_to_io(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'I', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
            elif t[0:1] == 'B':
                biluo[i] = t.replace('B', 'I', 1)
        return biluo 


    def subtokenize(self, sdoc, task, tokenizer):
        subtokens = []
        for token in sdoc.doc.tokens:
            begin = max(0, len(subtokens))
            for chunk in tokenizer.tokenize(token.text):
                subtokens.append(SubToken(token, chunk))
            end = max(0, len(subtokens))
            token.subtokens_interval = [begin, end]
        sdoc.subtokens = subtokens
        return sdoc


    def merge_discontinuous_overlaps(self, sdocs, annotation_type):
        LOG.info("Merge discontinuous overlaps")
        for sdoc in tqdm(sdocs):
            for span in sdoc.doc.spans:
                if span.contains_annotation(annotation_type):
                    for i in span.intervals:
                        for j in span.intervals:
                            if i != j and i.overlaps(j):
                                i.begin = min(i.begin, j.begin)
                                i.end = max(i.end, j.end)
                                span.intervals.remove(j)
        return sdocs


    def merge_overlaps(self, sdocs, annotation_type):
        LOG.info("Merge overlaps")
        for sdoc in tqdm(sdocs):
            for i in sdoc.doc.spans:
                for j in sdoc.doc.spans:
                    if i != j and i.contains_annotation(annotation_type) \
                              and j.contains_annotation(annotation_type) \
                              and len(i.intervals) == 1 \
                              and len(j.intervals) == 1:
                        if i != j and i.intervals[0].overlaps(j.intervals[0]):
                            i.intervals[0].begin = min(i.intervals[0].begin, j.intervals[0].begin)
                            i.intervals[0].end = max(i.intervals[0].end, j.intervals[0].end)
                            sdoc.doc.spans.remove(j)
        return sdocs


    def solve_discontinuous(self, sdocs, annotation_type):
        LOG.info("Solve discontinuous")
        for sdoc in tqdm(sdocs):
            for span in sdoc.doc.spans:
                if span.contains_annotation(annotation_type) and \
                   len(span.intervals) > 1:
                    new_spans = []
                    for interval in span.intervals:
                        new_spans.append(Span( document = span.document,
                                               document_id = span.document_id,
                                               annotations = span.annotations,
                                               intervals = [interval],
                                               tokens = span.document.tokens_in(interval) )) 
                    sdoc.doc.spans.remove(span)
                    sdoc.doc.spans.extend(new_spans)
        return sdocs