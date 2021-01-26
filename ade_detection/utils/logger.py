#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import logging 
import sys

from ade_detection.utils.env import Env


class Logger(object):

    @staticmethod
    def getLogger(name):

        log_level = logging.INFO

        if Env.get_value(Env.LOG_LEVEL) == 'CRITICAL':
            log_level = logging.CRITICAL 
        elif Env.get_value(Env.LOG_LEVEL) == 'ERROR':
            log_level = logging.ERROR  
        elif Env.get_value(Env.LOG_LEVEL) == 'WARNING':
            log_level = logging.WARNING  
        elif Env.get_value(Env.LOG_LEVEL) == 'INFO':
            log_level = logging.INFO  
        elif Env.get_value(Env.LOG_LEVEL) == 'DEBUG':
            log_level = logging.DEBUG  

        logging.basicConfig(level=log_level)

        LOG = logging.getLogger(name)
        #LOG.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
        return LOG