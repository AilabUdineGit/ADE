#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.domain.document import Document
from ade_detection.domain.enums import *


class SubDocument(object):


    def __init__(self, id: int, document_id: int, text: str, doc: Document,
                 tags: list = None, subtokens: list = None, 
                 num_tags: list = None, num_subtokens: list = None, attention_mask: list = None):
        self.id = id
        self.document_id = document_id
        self.subtokens = subtokens
        self.num_subtokens = num_subtokens
        self.tags = tags
        self.num_tags = num_tags
        self.attention_mask = attention_mask
        self.text = text
        self.doc = doc


    def copy(self):
        return SubDocument(self.id, self.document_id, self.text, self.doc, 
                           self.tags, self.subtokens, self.num_tags, self.num_subtokens,
                           self.attention_mask) 

    def __eq__(self, other):
        return self.id == other.id and \
               self.document_id == other.document_id and \
               self.subtokens == other.subtokens and \
               self.tags == other.tags and \
               self.num_tags == other.num_tags and \
               self.num_subtokens == other.num_subtokens and \
               self.attention_mask == other.attention_mask and \
               self.text == other.text and \
               self.doc == other.doc