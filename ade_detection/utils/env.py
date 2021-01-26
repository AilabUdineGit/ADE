#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from dotenv import load_dotenv
import json
import os 

import ade_detection.utils.localizations as loc


class Env(object):

    MAX_CHARS_IN_BETWEEN = 'MAX_CHARS_IN_BETWEEN' 
    LOG_LEVEL = 'LOG_LEVEL'
    MAX_LETTERS_IN_BETWEEN = 'MAX_LETTERS_IN_BETWEEN'
    TAC_SOURCE = 'TAC_SOURCE'
    TWIMED_SOURCE = 'TWIMED_SOURCE' 
    SPACY_MODEL = 'SPACY_MODEL' 
    DB = 'DB' 

    TWITTER = 'Twitter'
    LOCAL_STORAGE = 'LocalStorage'
    INTEGRATION_TEST_DB = 'INTEGRATION_TEST_DB'
    TEST_DB = 'TEST_DB'
    DEV_DB = 'DEV_DB'


    @staticmethod
    def load():
        '''setup environment from .env configs'''
        
        load_dotenv(os.path.join(os.path.abspath('.'), '.env'))


    @staticmethod
    def get_value(key: str) -> str:
        return os.environ[key]


    @staticmethod
    def set_value(key: str, value: str) -> None:
        os.environ[key] = value