#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from sqlalchemy import Column, ForeignKey, Integer, String, Text
from ade_detection.services import Base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey


class Attribute(Base):
    __tablename__ = 'attributes'
    id = Column(Integer, primary_key=True, autoincrement=True)

    document_id = Column(Integer, ForeignKey('documents.id'))
    document = relationship('Document', back_populates='attributes')

    key = Column(String)
    value = Column(String)


    def __eq__(self, other):
        return self.id == other.id and \
               self.document_id == other.document_id and \
               self.document == other.document and \
               self.key == other.key and \
               self.value == other.value