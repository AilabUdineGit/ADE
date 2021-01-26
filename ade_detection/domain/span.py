#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

from ade_detection.services import Base, spans_per_token
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.interval import Interval
from ade_detection.domain.token import Token


class Span(Base):
    __tablename__ = 'spans'
    id = Column(Integer, primary_key=True, autoincrement=True)

    document_id = Column(Integer, ForeignKey('documents.id'))
    document = relationship('Document', back_populates='spans')
    
    annotations = relationship('Annotation', order_by=Annotation.id, back_populates='span')
    intervals = relationship('Interval', order_by=Interval.id, back_populates='span')
    
    tokens = relationship('Token', secondary=spans_per_token,
                          back_populates='spans')


    def contains_annotation(self, annotation:str):
        return len(list(filter(lambda x: x.key == annotation, self.annotations))) > 0


    def contains_token(self, token:Token):
        return len(list(filter(lambda x: token.is_in(x), self.intervals))) > 0


    def __eq__(self, other):
        return self.id == other.id and \
               self.document_id == other.document_id and \
               self.document == other.document and \
               self.annotations == other.annotations and \
               self.intervals == other.intervals and \
               self.tokens == other.tokens