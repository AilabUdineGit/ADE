#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None

# load .env configs
from ade_detection.utils.env import Env
Env.load()
from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)

import os 
import sys
import shutil
from os import path

from ade_detection.services.tokenization_service import TokenizationService
from ade_detection.models.dirkson_final_trainer import DirksonFinalTrainer
from ade_detection.models.dirkson_task_loader import DirksonTaskLoader
from ade_detection.importer.smm4h19_importer import SMM4H19Importer
from ade_detection.services.database_service import DatabaseService
from ade_detection.models.bert_task_loader import BertTaskLoader
from ade_detection.models.dirkson_trainer import DirksonTrainer
from ade_detection.importer.cadec_importer import CadecImporter
from ade_detection.services.model_service import ModelService
from ade_detection.domain.train_config import TrainConfig
from ade_detection.models.bert_trainer import BertTrainer
from ade_detection.models.comparator import Comparator
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.task import Task
from ade_detection.domain.enums import *

import numpy as np
import torch
import random


class CliHandler(object):

    '''Cli business logic, given the arguments typed
    calls the right handlers/procedures of the pipeline'''


    def __init__(self, args):
        try:
            if not path.exists(loc.abs_path([loc.TMP, loc.BIO_BERT_GIT])):
                ModelService.get_bio_git_model()
        except:
            pass
        if args.import_ds:
            self.import_handler()
        elif args.run is not None:
            self.run_handler(args.run[0])
        elif args.clean:
            self.clean_handler()
        else:
            self.default_handler()


    # Command Handlers 

    def default_handler(self):
        LOG.info('Welcome to ADE Detection script! Type -h for help \n\n' +
                 '[You have to type at least one command of the pipeline]\n')


    def clean_handler(self):
        LOG.info('Clean temporary files')
        fm.rmdir(loc.TMP_PATH)

    
    def import_handler(self):
        if os.path.exists(loc.DB_PATH):
            os.remove(loc.DB_PATH)
        LOG.info('DB creation')
        DB = DatabaseService()
        DB.create_all()
        
        LOG.info('Import CADEC')
        CadecImporter()
        LOG.info('Import SMM4H-19')
        SMM4H19Importer()
        LOG.info('Tokenization')
        TokenizationService(CORPUS.CADEC)
        TokenizationService(CORPUS.SMM4H19_TASK2)


    def set_all_seed(self, seed):
        LOG.info(f"random seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # if you are using GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    def run_handler(self, run_path):
        json = fm.from_json(loc.abs_path([loc.ASSETS, loc.RUNS, run_path]))
        for task in json:
#            try:
                random_seed = int(task['train_config']['random_seed'])
                self.set_all_seed(random_seed)

                train_config = TrainConfig(int(task['train_config']['max_patience']),
                                           float(task['train_config']['learning_rate']), 
                                           float(task['train_config']['dropout']),
                                           int(task['train_config']['epochs']), 
                                           random_seed, 
                                           float(task['train_config']['epsilon']))

                task = Task( task['id'], task['split_folder'], 
                             enums_by_list(TIDY_MODE, task['tidy_modes']), 
                             enum_by_name(CORPUS, task['corpus']), 
                             enum_by_name(NOTATION, task['notation']), 
                             enum_by_name(MODEL, task['model']), 
                             enum_by_name(ARCHITECTURE, task['architecture']), 
                             enums_by_list(ANNOTATION_TYPE, task['goal']), 
                             enum_by_name(TRAIN_MODE, task['train_mode']), 
                             train_config )
                
                model = task.model
                if model == MODEL.DIRKSON:
                    self.dirkson_task_handler(task)
                else: 
                    self.bert_task_handler(task)
#            except:
#                LOG.warning('Task Failed')
        

    def dirkson_task_handler(self, task):
        DirksonTaskLoader(task)
        DirksonTrainer(task.corpus.name)
        DirksonFinalTrainer(task.corpus.name)
        Comparator.compare_dirkson(task.corpus.name)
    

    def bert_task_handler(self, task):
        loaded_task = BertTaskLoader(task)
        BertTrainer(loaded_task.task) 


# Used when spawned in a new process

if __name__ == '__main__':
    LOG.info(f'Subprocess started {sys.argv}')
    sys.stdout.flush()
    from ade_detection.cli import Parser
    args = Parser().parse()    
    CliHandler(args)