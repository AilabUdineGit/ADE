#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import os
import json


'''Localization keys and files'''


def abs_path(relative_path: list) -> str:

    ''' given an array of localizations (relative path) 
    returns and absolute path starting from cwd '''

    abs_path = os.getcwd()
    for p in relative_path:
        abs_path = os.path.join(abs_path, p)
    if os.path.isdir(abs_path) and not os.path.exists(abs_path):
        os.mkdir(abs_path)
    return abs_path


# Localizations for files

CADEC_ZIP = 'CADEC.v2.zip'
BIO_BERT_ZIP = 'biobert.zip'
SMM4H19_ZIP = 'SMM4H_2019_task2_dataset.zip'
DIRKSON_TEST_RESULTS_PICKLE = 'dirkson_test_results.pickle'
DIRKSON_VALIDATION_TXT = 'dirkson_validation.txt'
DIRKSON_TEST_TXT = 'dirkson_test.txt'
LOSS_TSV = 'loss.tsv'
WEIGHTS_TXT = 'weights.txt'


# Localizations for links

BIO_BERT_GIT_LINK = 'https://github.com/vthost/biobert-pretrained-pytorch/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.zip'


# Localizations for folders

ASSETS = 'assets'
TMP = 'tmp'
DATASETS = 'datasets'
MODELS = 'models'
SPLITS = 'splits'
RUNS = 'runs'
OBJ_LEX = 'obj_lex'

BIO_BERT_GIT = 'biobert_v1.1_pubmed'

SMM4H19 = 'smm4h19'
CADEC_ARCHIVE = 'cadec'
SMM4H19_ARCHIVE = 'SMM4H_2019_task2_dataset'
CLEAN_SMM4H19_CSV = 'SMM4H19.csv'
CADEC = 'cadec'
MEDDRA = 'meddra'
ORIGINAL = 'original'
SCT = 'sct'
TEXT = 'text'

DIRKSON = 'dirkson'

TWIMED_GOLD_CONFLATED = 'gold_conflated'
TWITTER = 'twitter'
PUBMED = 'pubmed'
TWIMED_TWITTER_TEXTS = 'twimed_twitter_texts'
TRAIN_DATA_1 = 'TrainData1.tsv'
TRAIN_DATA_2 = 'TrainData1.tsv'
TRAIN_DATA_3 = 'TrainData1.tsv'
TRAIN_DATA_4 = 'TrainData1.tsv'


# Localizations for paths

CADEC_ZIP_PATH = abs_path([ASSETS, DATASETS, CADEC, CADEC_ZIP])
SMM4H19_ZIP_PATH = abs_path([ASSETS, DATASETS, SMM4H19, SMM4H19_ZIP])

CADEC_ARCHIVE_PATH = abs_path([TMP, CADEC_ARCHIVE])
SMM4H19_ARCHIVE_PATH = abs_path([TMP, SMM4H19_ARCHIVE])
TMP_PATH = abs_path([TMP])

SMM4H19_TRAIN = 'TrainData{0}.tsv'
SMM4H19_TRAIN_PATH_1 = abs_path([TMP, SMM4H19_TRAIN.format(1)])
SMM4H19_TRAIN_PATH_2 = abs_path([TMP, SMM4H19_TRAIN.format(2)])
SMM4H19_TRAIN_PATH_3 = abs_path([TMP, SMM4H19_TRAIN.format(3)])
SMM4H19_TRAIN_PATH_4 = abs_path([TMP, SMM4H19_TRAIN.format(4)])

SMM4H19_CLEAN_TRAIN_PATH = abs_path([ASSETS, DATASETS, SMM4H19, CLEAN_SMM4H19_CSV])

CADEC_TEXTS_QUERY = abs_path([TMP, CADEC, TEXT, '*.txt'])


# Connection Strings

DB_CONNECTION_STRING = 'sqlite:///assets/db.sqlite'
TEST_DB_CONNECTION_STRING = 'sqlite:///tmp/test_db.sqlite'
INTEGRATION_TEST_DB_CONNECTION_STRING = 'sqlite:///tmp/integration_test_db.sqlite'
DB = 'db.sqlite'
TEST_DB = 'test_db.sqlite'
INTEGRATION_DB = 'integration_test_db.sqlite'
DB_PATH = abs_path([ASSETS, DB])
TEST_DB_PATH = abs_path([TMP, TEST_DB])
INTEGRATION_DB_PATH = abs_path([TMP, INTEGRATION_DB])


# Localizations for exceptions

FAIL_ENGINE_CREATION_EXCEPTION = 'Fail to create db engine' 


# Splits

TEST_ID = 'test.id'
TRAIN_ID = 'train.id'
VALIDATION_ID = 'validation.id'

CADEC_SPLIT = 'cadec'
SMM4H19_SPLIT = 'smm4h19_task2'
SMM4H19_SPLIT_PATH = abs_path([ASSETS, SPLITS, SMM4H19_SPLIT])