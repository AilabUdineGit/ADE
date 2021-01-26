from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, Integer, String, Text, Table
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

spans_per_token = Table('spans_per_token', Base.metadata,
    Column('span_id', Integer, ForeignKey('spans.id')),
    Column('token_id', Integer, ForeignKey('tokens.id'))
)

annotations_per_token = Table('annotations_per_token', Base.metadata,
    Column('annotation_id', Integer, ForeignKey('annotations.id')),
    Column('token_id', Integer, ForeignKey('tokens.id'))
)