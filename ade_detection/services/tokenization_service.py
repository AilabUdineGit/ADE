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
from os import path

from ade_detection.services.database_service import DatabaseService
from ade_detection.services.spacy_service import SpacyService
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.sentence import Sentence
from ade_detection.domain.interval import Interval
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.token import Token
from ade_detection.domain.span import Span
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env
import re


class TokenizationService(object):

    '''Tokenization script
    '''

    def __init__(self, dataset: CORPUS):  
        self.spacy = SpacyService()
        db = DatabaseService()
        session = db.new_session()
        docs = session.query(Document) \
                      .filter_by(corpus = dataset) \
                      .all()
        LOG.info('Tokenization and sentencization in progress...')
        for doc in tqdm(docs):

            for span in doc.spans:
                for annotation in span.annotations:
                    if annotation.key == ANNOTATION_TYPE.ADR: 
                        if len(span.intervals) == 1:
                            interval = span.intervals[0]
                            if doc.text[interval.begin: interval.end].lower() != annotation.value.lower():
                                begin = self.nearest([m.start() for m in re.finditer(re.escape(annotation.value.lower()), doc.text.lower())], interval.begin)
                                if begin is None:
                                    continue
                                interval.begin = begin
                                interval.end = begin + len(annotation.value)

            tokens = self.tokenizer(doc.text)
            doc.tokens = self.tokens_postprocessing(tokens)
            doc.sentences = self.sentencizer(doc.text)
            doc = self.calibrate_intervals(doc)
            doc = self.update_annotations_per_token(doc)
        LOG.info('Tokenization and sentencization completed successfully!')
        session.commit()
        LOG.info('DB updated successfully!')


    def nearest(self, array, value):
        if len(array) > 0:
            winner = array[0]
            best_delta = abs(array[0] - value)
            for i in array:
                delta = abs(i - value)
                if delta < best_delta:
                    winner = i 
            return winner
        else:
            return None


    def calibrate_intervals(self, doc:Document):
        for span in doc.spans:
            for interval in span.intervals:
                tokens = doc.tokens_touched(interval)
                assert len(tokens) > 0
                if tokens[0].begin != interval.begin:
                    interval.begin = tokens[0].begin 
                if tokens[-1].end != interval.end:
                    interval.end = tokens[-1].end 
        return doc


    def tokenizer(self, text:str):
        tokens = []
        for t in self.spacy.tokenizer(text):
            tokens.append(Token(begin = t.idx, 
                                end = t.idx + len(t.text), 
                                text = t.text))
        return tokens


    def sentencizer(self, text:str):
        sentences = []
        for sent in self.spacy.sentencizer(text):
            sentence = sent.string.strip()
            sentences.append(Sentence(begin = sent.start_char, 
                                      end = sent.start_char + len(sentence)))
        return sentences


    def tokens_postprocessing(self, tokens):
        for t in tokens:
            if self.is_number(t.text):
                t.text = 'number'
            elif self.is_hashtag(t.text):
                t.text = t.text[1:len(t.text)]
            elif self.is_username(t.text):
                t.text = 'username'
            elif self.is_link(t.text):
                t.text = 'link'
            else:
                t.text = self.delete_triples(t.text)
        return tokens


    def update_annotations_per_token(self, doc:Document):
        for span in doc.spans:
            for interval in span.intervals:
                interval_tokens = doc.tokens_in(interval) 
                span.tokens.extend(interval_tokens)
            for annotation in span.annotations:
                annotation.tokens.extend(span.tokens)


    def is_number(self, token):
        return token.replace('.','',1).replace(',','',1).isdigit()


    def is_hashtag(self, token):
        if len(token) > 0:
            return token[0] == '#' and len(token) > 1
        else:
            return False

    def is_username(self, token):
        if len(token) > 0:
            return token[0] == '@' and len(token) > 1
        else:
            return False


    def is_link(self, token):
        if len(token) > 4:
            return token[0:4] == 'http' or token[0:3] == 'www' 
        else:
            return False


    def delete_triples(self, token):
        i = 0
        token = list(token)
        while i < len(token) - 2:
            if token[i] == token[i+1] and token[i+1] == token[i+2] and token[i].isalpha():
                del(token[i])
                i -= 1   
            i += 1            
        return "".join(token)
