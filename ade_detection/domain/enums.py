#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import ade_detection.utils.localizations as loc
import enum


def enum_by_name(enum, name: str):
    for i in enum:
        if i.name == name:
            return i 
    return None


def enums_by_list(enum, array: list):
    res = []
    for i in array:
        res.append(enum_by_name(enum, i))
    return res


class CORPUS(enum.Enum):
    CADEC = 1
    TAC = 2
    TWIMED_TWITTER = 3
    TWIMED_PUBMED = 4
    SMM4H19_TASK1 = 5
    SMM4H19_TASK2 = 6
    PSY_TAR = 7
    BIO_SCOPE = 8


class PARTITION_TYPE(enum.Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class NOTATION(enum.Enum):
    IO = 1
    IOB = 2
    BILUO = 3


class MODEL(str, enum.Enum):
    SPAN_BERT_CASED = 'SpanBERT/spanbert-base-cased'
    BIO_BERT = 'dmis-lab/biobert-v1.1'
    BIO_BERT_GIT = loc.abs_path([loc.TMP, loc.BIO_BERT_GIT])
    SCI_BERT = 'allenai/scibert_scivocab_cased'
    BIO_CLINICAL_BERT = 'emilyalsentzer/Bio_ClinicalBERT'
    BERT_TWEET = 'vinai/bertweet-base'
    PUB_MED_BERT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    BASE_UNCASED = 'bert-base-uncased'
    DIRKSON = 'dirkson'
    

class ARCHITECTURE(enum.Enum):
    BERT_WRAPPER = 1
    BERT_CRF = 2
    BERT_LSTM = 3


class TIDY_MODE(enum.Enum):
    MERGE_OVERLAPS = 1
    MERGE_ADJACENT = 2
    SOLVE_DISCONTINUOUS = 3


class TRAIN_MODE(enum.Enum):
    VALIDATION = 1
    TESTING = 2
    JUST_TESTING = 3


class ANNOTATION_TYPE(enum.Enum):
    ADR = 1
    Drug = 2
    Disease = 3
    Indication = 4
    Symptom = 5
    Finding = 6
    related_drug = 7
    target_drug = 8
    meddra_code = 9
    meddra_term = 10


def annotation_by_name(name:str):
    for annotation in ANNOTATION_TYPE:
        if annotation.name == name:
            return annotation 
    raise ValueError("Unknown annotation_type: " + name)


BATCH_SIZE = { CORPUS.CADEC : 8, 
               CORPUS.TAC : 32, 
               CORPUS.TWIMED_TWITTER : 32, 
               CORPUS.TWIMED_PUBMED : 8, 
               CORPUS.SMM4H19_TASK2 : 32 }


MAX_SEQ_LEN = { CORPUS.CADEC : 512, 
                CORPUS.TAC : 64, 
                CORPUS.TWIMED_TWITTER : 64, 
                CORPUS.TWIMED_PUBMED : 512, 
                CORPUS.SMM4H19_TASK2 : 64 }


class COMPARATOR_MODE(enum.Enum):
    STRICT = 1
    PARTIAL = 2
