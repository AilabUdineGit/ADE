#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.domain.enums import *
from ade_detection.domain.token import Token


class SubToken(object):


    def __init__(self, token: Token, text: str):
        self.token = token
        self.text = text
           

    def __eq__(self, other):
        return self.token == other.token and \
               self.text == other.text 