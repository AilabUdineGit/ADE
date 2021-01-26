#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.services import Base, spans_per_token, annotations_per_token
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from ade_detection.domain.interval import Interval
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey


class Token(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True, autoincrement=True)

    document_id = Column(Integer, ForeignKey('documents.id'))
    document = relationship('Document', back_populates='tokens')

    begin = Column(Integer)
    end = Column(Integer)
    text = Column(Text)

    spans = relationship('Span',
                         secondary=spans_per_token,
                         back_populates='tokens')

    annotations = relationship('Annotation',
                               secondary=annotations_per_token,
                               back_populates='tokens')


    def is_touched(self, interval:Interval):
        t1 = self.begin
        t2 = self.end 
        i1 = interval.begin 
        i2 = interval.end
        return (t1 >= i1 and t1 < i2) or \
               (t2 > i1 and t2 <= i2) or \
               (i1 >= t1 and i1 < t2) or \
               (i2 > t1 and i2 <= t2)


    def is_in(self, interval:Interval):
        t1 = self.begin
        t2 = self.end 
        i1 = interval.begin 
        i2 = interval.end
        return t1 >= i1 and t2 <= i2


    def __eq__(self, other):
        return self.id == other.id and \
               self.document_id == other.document_id and \
               self.document == other.document and \
               self.spans == other.spans and \
               self.begin == other.begin and \
               self.end == other.end and \
               self.text == other.text 