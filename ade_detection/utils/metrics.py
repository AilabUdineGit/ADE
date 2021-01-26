#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


class Metrics(object):

    @staticmethod
    def get_f1score(p,r):
        # avoid division by zero
        if p+r != 0:
            return 2 * (p*r)/(p+r)
        else:
            return 0


    # @input training_loss, validation_loss
    # @output True if tr_loss is less than one order of magnitude wrt vl_loss
    #         False otherwise
    @staticmethod
    def overfit(tr_loss, vl_loss):
        #print('valid/train: ' + str(vl_loss/tr_loss))
        return vl_loss/tr_loss >= 10