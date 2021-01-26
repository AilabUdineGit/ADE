#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import unittest

from ade_detection.cli_handler import CliHandler
from ade_detection.cli import Parser


class RunSMM4HTest(unittest.TestCase):
    '''Test a single run on SMM4H'''


    def test(self):
        command = '--run single_run_smm4h.json'.split()
        args = Parser().parse_args(command)    
        CliHandler(args)
