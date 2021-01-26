#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from flair.data_fetcher import NLPTaskDataFetcher
from flair.visual.training_curves import Plotter
from flair.embeddings import StackedEmbeddings
from flair.embeddings import TokenEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import WordEmbeddings
from flair.embeddings import BertEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data_fetcher import NLPTask
from typing import List
import pandas as pd
from os import path 
import os

from ade_detection.domain.enums import *
import ade_detection.utils.localizations as loc


class DirksonTrainer(object):

    def __init__(self, corpus_name:str):

        corpus = NLPTaskDataFetcher.load_column_corpus( loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON]), 
                                                        {0: 'text', 1: 'ner'}, 
                                                        train_file=corpus_name + loc.DIRKSON_VALIDATION_TXT,
                                                        test_file=corpus_name + loc.DIRKSON_TEST_TXT )

        embedding_types = [
            BertEmbeddings('bert-base-uncased'),
            FlairEmbeddings('mix-forward'),
            FlairEmbeddings('mix-backward')
        ]

        tag_type = 'ner'
        embeddings = StackedEmbeddings(embeddings=embedding_types)
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True)

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        if not path.exists:
            os.mkdir(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name]))
        trainer.train( loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name]),
                       learning_rate=0.1,
                       mini_batch_size=32,
                       max_epochs=150 )

        plotter = Plotter()
        plotter.plot_training_curves(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name, loc.LOSS_TSV]))
        plotter.plot_weights(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name, loc.WEIGHTS_TXT]))