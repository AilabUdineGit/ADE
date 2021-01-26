#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import pandas as pd

import ade_detection.utils.localizations as loc
from ade_detection.domain.enums import *


class DirksonFinalTrainer(object):

    def __init__(self, corpus_name:str):
        f = open(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, 
                               corpus_name + loc.DIRKSON_TEST_TXT]), "r", encoding="utf8")

        all_doc_text = []
        all_labels = []

        temp_text = ""
        temp_labels = []
        for line in f:
            if line == "\n":
                all_doc_text.append(temp_text[1:])
                all_labels.append(temp_labels)
                temp_text = ""
                temp_labels = []
            else:
                temp_text = temp_text + " " + line[0:-3]
                temp_labels.append(line[-2:-1])

        
        model = SequenceTagger.load_from_file(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name, "best-model.pt"]))

        doc_text = []
        truth_labels = []
        pred_labels = []

        for text, labels in tqdm(zip(all_doc_text, all_labels), total=len(all_doc_text)):
            sentence = Sentence(text)
            model.predict(sentence)
            pred = self.fromTaggedToIOB(sentence.to_tagged_string())
            doc_text.append(text.split(" "))
            truth_labels.append(labels)
            pred_labels.append(pred)

            if len(pred) != len(labels):
                print(text)
                print(sentence)


        df = pd.DataFrame({ 'doc_text': doc_text,
                            'labels': truth_labels,
                            'pred_labels': pred_labels })

        df.to_pickle(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, 
                                   corpus_name + loc.DIRKSON_TEST_RESULTS_PICKLE]))


    def fromTaggedToIOB(self, pred):

        model_prediction = []

        pred_list = pred.split(" ")
        n = len(pred_list)
        last_idx = n-1

        while last_idx >= 0:
            if pred_list[last_idx] == "<B>":
                model_prediction.insert(0, "B")
                last_idx = last_idx-2
            elif pred_list[last_idx] == "<I>":
                model_prediction.insert(0, "I")
                last_idx = last_idx-2
            else: # ovvero <O>
                model_prediction.insert(0, "O")
                last_idx = last_idx-1
        return model_prediction

