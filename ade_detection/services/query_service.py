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


class QueryService(object):

    '''Query script
    '''

    def __init__(self, session = None):       
        self.db = DatabaseService()
        if session is None:
            self.session = self.db.new_session()
        else:       
            self.session = session


    def docs_by_corpus(self, corpus:CORPUS):
        return self.session.query(Document) \
                   .filter_by(corpus = corpus) \
                   .all()


    def doc_by_external_id(self, docs:list, external_id:str):
        return next(filter(lambda x: x.external_id == external_id, docs), None)


    def doc_subtokens(self, doc:Document):
        return self.session.query(Document) \
                   .filter_by(corpus = corpus) \
                   .all()