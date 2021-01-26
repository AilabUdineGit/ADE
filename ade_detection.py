#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


# load .env configs
from ade_detection.utils.env import Env
Env.load()

# ade_detection/cli.py wrapper
import subprocess 
import os 
import sys 

from ade_detection.cli import Parser
from ade_detection.cli_handler import CliHandler

args = Parser().parse()    
CliHandler(args)