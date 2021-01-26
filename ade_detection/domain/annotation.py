#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from sqlalchemy import Column, ForeignKey, Integer, String, Text, Enum
from ade_detection.services import Base, annotations_per_token
from ade_detection.domain.enums import *
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey


class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True, autoincrement=True)

    span_id = Column(Integer, ForeignKey('spans.id'))
    span = relationship('Span', back_populates='annotations')

    tokens = relationship('Token', secondary=annotations_per_token,
                          back_populates='annotations')

    key = Column(Enum(ANNOTATION_TYPE))
    value = Column(String)

    def __eq__(self, other):
        return self.id == other.id and \
               self.span_id == other.span_id and \
               self.span == other.span and \
               self.key == other.key and \
               self.value == other.value