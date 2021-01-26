#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from pandas.core.frame import DataFrame
from tqdm import tqdm
import pandas as pd 
import numpy as np
from os import path
import os

from ade_detection.services.database_service import DatabaseService
from ade_detection.services.query_service import QueryService
from ade_detection.domain.split_config import SplitConfig
from ade_detection.domain.subdocument import SubDocument
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.sentence import Sentence
from ade_detection.domain.interval import Interval
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.token import Token
from ade_detection.domain.split import Split
from ade_detection.domain.span import Span
from ade_detection.domain.task import Task
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class SplitService(object):

    '''Split script
    '''

    def __init__(self):        
        self.db = DatabaseService()     
        self.session = self.db.new_session()
        self.query = QueryService(self.session)


    def split(self, split:SplitConfig):
        docs = self.query.docs_by_corpus(split.dataset)

        (train_ids, test_ids, validation_ids) = self.new_split(docs, split)

        if not path.exists(split.path):
            os.mkdir(split.path)
        fm.to_id(train_ids, path.join(split.path, loc.TRAIN_ID))
        fm.to_id(test_ids, path.join(split.path, loc.TEST_ID))
        fm.to_id(validation_ids, path.join(split.path, loc.VALIDATION_ID))

        LOG.info('split saved successfully!')


    def new_split(self, docs, split):
        LOG.info('dataset split in progress...')
        adr_docs = list(filter(lambda x: x.contains_annotation('ADR'), docs))
        no_adr_docs = list(filter(lambda x: not x.contains_annotation('ADR'), docs))

        (adr_train, adr_test, adr_validation) = self.split_dataset(adr_docs, split)
        (no_adr_train, no_adr_test, no_adr_validation) = self.split_dataset(no_adr_docs, split)

        train_docs = adr_train + no_adr_train
        test_docs = adr_test + no_adr_test
        validation_docs = adr_validation + no_adr_validation

        LOG.info('dataset splitted successfully!')
        return ([x.external_id for x in train_docs], 
                [x.external_id for x in test_docs],
                [x.external_id for x in validation_docs])


    def split_dataset(self, docs, split):
        validation_end = len(docs)
        train_end = int(validation_end * split.train_percentage)
        test_end = train_end + int(validation_end * split.test_percentage)

        return (docs[0:train_end], docs[train_end:test_end], docs[test_end:validation_end])

    
    def load_split(self, task:Task):
        split_directory = loc.abs_path([loc.ASSETS, loc.SPLITS, task.split_folder])
        train = fm.from_id(path.join(split_directory, loc.TRAIN_ID))
        test = fm.from_id(path.join(split_directory, loc.TEST_ID))
        validation = fm.from_id(path.join(split_directory, loc.VALIDATION_ID))

        docs = self.query.docs_by_corpus(task.corpus)
        train_subdocs = self.load_partition(train, docs, 0)
        test_subdocs = self.load_partition(test, docs, train_subdocs[-1].id + 1)
        validation_subdocs = self.load_partition(validation, docs, test_subdocs[-1].id + 1)
        return Split(train_subdocs, test_subdocs, validation_subdocs)


    def load_partition(self, ids, docs, offset):
        subdocs = []
        for index, id in enumerate(ids):
            doc = self.query.doc_by_external_id(docs, id)
            if doc is not None:
                subdocs.append(SubDocument(offset + index,  # id
                                        doc.id,          # document_id
                                        doc.text,        # text
                                        doc ))           # doc
        return subdocs