#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


# load .env configs
from ade_detection.utils.env import Env
Env.load()

Env.set_value(Env.DB, Env.INTEGRATION_TEST_DB)

# load tests
import unittest

from integration_tests import ImportTest
from integration_tests import RunCadecTest
from integration_tests import RunSMM4HTest
from integration_tests import RunTMRLTest


'''Run all tests in integration_tests/'''

unittest.main()