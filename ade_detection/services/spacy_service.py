#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
import zipfile
import pandas as pd
import numpy as np
import tweepy
import os
from spacy.lang.en import English
import spacy
import re

from ade_detection.utils.env import Env


class SpacyService(object):


    def __init__(self):
        spacy_model = Env.get_value(Env.SPACY_MODEL)
        if spacy_model == 'english':
            self.nlp = English()
        else:
            self.nlp = spacy.load(spacy_model) 
        self.nlp.add_pipe(self.hashtag_pipe)
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))


    def hashtag_pipe(self, doc):
        '''Inspired by https://github.com/explosion/spaCy/issues/503
        '''
        i = 0
        while i < len(doc) - 1: 
            token = doc[i]
            if token.text == '#':
                if re.match(r'^\w+$', str(doc[i+1])):
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[i:i+2])
            i += 1
        return doc


    def tokenizer(self, text:str):
        return self.nlp(text)


    def sentencizer(self, text:str):
        return self.nlp(text).sents