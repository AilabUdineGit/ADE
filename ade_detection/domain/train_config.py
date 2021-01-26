#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.domain.enums import *


class TrainConfig(object):


    def __init__(self, max_patience: int, learning_rate: float, dropout: float,
                 epochs: int, random_seed: float, epsilon: float = 1e-8):
        self.max_patience = max_patience
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.epsilon = epsilon
        self.random_seed = random_seed
           

    def __eq__(self, other):
        return self.max_patience == other.max_patience and \
               self.learning_rate == other.learning_rate and \
               self.dropout == other.dropout and \
               self.epochs == other.epochs and \
               self.epsilon == other.epsilon and \
               self.random_seed == other.random_seed 