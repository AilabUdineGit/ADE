#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from ade_detection.services import Base
from sqlalchemy import ForeignKey


class Interval(Base):
    __tablename__ = 'intervals'
    id = Column(Integer, primary_key=True, autoincrement=True)

    span_id = Column(Integer, ForeignKey('spans.id'))
    span = relationship('Span', back_populates='intervals')

    begin = Column(Integer)
    end = Column(Integer)


    def overlaps(self, other):
        x1 = self.begin
        x2 = self.end 
        y1 = other.begin 
        y2 = other.end
        return (x1 >= y1 and x1 <= y2) or \
               (x2 > y1 and x2 <= y2) or \
               (y1 >= x1 and y1 <= x2) or \
               (y2 >= x1 and y2 <= x2)
               

    def __eq__(self, other):
        return self.id == other.id and \
               self.span_id == other.span_id and \
               self.span == other.span and \
               self.begin == other.begin and \
               self.end == other.end 