#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from sqlalchemy import Column, ForeignKey, Integer, String, Text, Enum
from sqlalchemy.orm import relationship
from ade_detection.services import Base

from ade_detection.domain.attribute import Attribute
from ade_detection.domain.sentence import Sentence
from ade_detection.domain.interval import Interval
from ade_detection.domain.token import Token
from ade_detection.domain.span import Span
from ade_detection.domain.enums import *


class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    external_id = Column(String)
    text = Column(Text)
    corpus = Column(Enum(CORPUS))

    sentences = relationship('Sentence', order_by=Sentence.id, back_populates='document')
    tokens = relationship('Token', order_by=Token.id, back_populates='document')
    spans = relationship('Span', order_by=Span.id, back_populates='document')
    attributes = relationship('Attribute', order_by=Attribute.id, back_populates='document')


    def contains_annotation(self, annotation:str):
        return len(list(filter(lambda x: x.contains_annotation(annotation), self.spans))) > 0


    def tokens_touched(self, interval: Interval):
        return list(filter(lambda x: x.is_touched(interval), self.tokens))


    def tokens_in(self, interval: Interval):
        return list(filter(lambda x: x.is_in(interval), self.tokens))


    def __eq__(self, other):
        return self.id == other.id and \
               self.external_id == other.external_id and \
               self.text == other.text and \
               self.corpus == other.corpus and \
               self.sentences == other.sentences and \
               self.tokens == other.tokens and \
               self.spans == other.spans