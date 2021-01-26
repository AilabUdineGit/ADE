#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import subprocess
import argparse
import sys
import os 

from ade_detection.cli_handler import CliHandler


class Parser(object):
    '''Cli entry point of the script, based on the library argparse
    see also: https://docs.python.org/3.9/library/argparse.html'''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter, 
            description = '''                              Welcome to ADE Detection Script :)  

+---------------------------------------------------------------------------------------------+
|   /  _  \ \______ \ \_   _____/ \______ \   _____/  |_  ____   _____/  |_|__| ____   ____   |
|  /  /_\  \ |    |  \ |    __)_   |    |  \_/ __ \   __\/ __ \_/ ___\   __\  |/  _ \ /    \  |
| /    |    \|    `   \|        \  |    `   \  ___/|  | \  ___/\  \___|  | |  (  <_> )   |  \ | 
| \____|__  /_______  /_______  / /_______  /\___  >__|  \___  >\___  >__| |__|\____/|___|  / |
|         \/        \/        \/          \/     \/          \/     \/                    \/  |
+---------------------------------------------------------------------------------------------+''')


        self.parser.add_argument('-i', '--import', dest='import_ds', action='store_const',
                                const=True, default=False,
                                help='drop database and import all datasets')


        self.parser.add_argument('-c', '--clean', dest='clean', action='store_const',
                                const=True, default=False,
                                help='clean temporary/useless files')


        self.parser.add_argument('--run', dest='run', metavar='N', type=str, nargs=1,
                                help='run an array of tasks (remember to specify the name of your run .json)')


    def parse(self):    
        return self.parser.parse_args()


    def parse_args(self, command):    
        return self.parser.parse_args(command)