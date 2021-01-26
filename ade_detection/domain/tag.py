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


class Tag(Base):
    __tablename__ = 'tags'
    id = Column(Integer, primary_key=True, autoincrement=True)

    task_id = Column(Integer, ForeignKey('tasks.id'))
    task = relationship('Task')

    document_id = Column(Integer, ForeignKey('document.id'))
    document = relationship('Document')

    tag = Column(String)


    def __eq__(self, other):
        return self.id == other.id and \
               self.task_id == other.task_id and \
               self.task == other.task and \
               self.document_id == other.document_id and \
               self.document == other.document