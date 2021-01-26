#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pandas.core.frame import DataFrame
from nltk import word_tokenize, pos_tag
from tqdm import tqdm
import pandas as pd 
import re

from ade_detection.models.dirkson_normalizer import DirksonNormalizer
from ade_detection.models.base_task_loader import BaseTaskLoader
from ade_detection.services.split_service import SplitService
from ade_detection.services.model_service import ModelService
from ade_detection.domain.task import Task
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class DirksonTaskLoader(BaseTaskLoader):

    '''Importer script for the Dirkson model'''

    def __init__(self, task: Task):
        super(DirksonTaskLoader, self).__init__(task)
        LOG.info('dataset export in progress...')

        split_svc = SplitService()
        self.task.split = split_svc.load_split(self.task)
        
        train = self.load(task.split.train)
        validation = self.load(task.split.validation)
        train_full = train + validation
        test = self.load(task.split.test)

        all_doc_text = [[y.text for y in x.doc.tokens] for x in train_full]
        all_labels = [x.tags for x in train_full]
        prep_text, prep_labels = self.preprocess_text(all_doc_text, all_labels)
        self.save(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, 
                                task.corpus.name + loc.DIRKSON_VALIDATION_TXT]), prep_text, prep_labels)

        all_doc_text = [[y.text for y in x.doc.tokens] for x in test]
        all_labels = [x.tags for x in test]
        prep_text, prep_labels = self.preprocess_text(all_doc_text, all_labels)
        self.save(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, 
                                task.corpus.name + loc.DIRKSON_TEST_TXT]), prep_text, prep_labels)
        
        LOG.info('dataset exported successfully!...')


    def load(self, sdocs):
        for annotation_type in self.task.goal:
            sdocs = self.merge_discontinuous_overlaps(sdocs, annotation_type)
            sdocs = self.solve_discontinuous(sdocs, annotation_type)
            sdocs = self.merge_overlaps(sdocs, annotation_type)
        for sdoc in sdocs:
            sdoc = self.tokens_single_biluo_tagging(sdoc, self.task)
            if self.task.notation == NOTATION.IOB:
                sdoc.tags = self.biluo_to_iob(sdoc.tags)
            if self.task.notation == NOTATION.IO:
                sdoc.tags = self.biluo_to_io(sdoc.tags)       
        return sdocs


    def remove_punc(self, post): 

        '''remove punctuation'''
        
        temp = []
        for word in post: 
            if re.fullmatch (r'[^\w\s]', word) == None: 
                temp.append(word)
            else: 
                pass
        return temp 


    def post_filter_char(self, msg):
        
        '''remove special chars'''

        final1 = msg.replace('Â', '')
        final2= final1.replace('â€™', '')
        final3 = final2.replace('â€œ', '')
        final4 = final3.replace('â€“', '')
        final5 = final4.replace('â€¦', '')
        final6 = final5.replace('â€', '')
        final7 = final6.replace('...', ' ')
        final8 = final7.replace ('`', '')
        final9 = final8.replace ('ðÿ˜', '')
        final10 = final9.replace ('¡', '')
        final11 = final10.replace ('©', '')
        final12 = final11.replace ('👀🙄', '')
        final13 = final12.replace ( '�', '')
        final14 = final13.replace ('💩', '')
        final15 = re.sub(r'(@ ?[a-zA-Z0-9-_]+[\.: ]?)', '', final14)
        return final15


    def preprocess_text(self, all_doc_text, all_labels):

        d = TreebankWordDetokenizer()
        preprocessed_text = []
        fixed_labels = []

        for text, labels in tqdm(zip(all_doc_text, all_labels), total=len(all_labels)):
            # iter on each word in the sentence

            temp_text = []
            temp_labels = []

            for word, label in zip(text, labels):

                # FASE 1 - normalize word
                temp_w = " " + word + " "
                norm_w = DirksonNormalizer().normalize([temp_w])

                # FASE 2 - remove punctuation
                punt_w = [self.remove_punc(m) for m in norm_w]

                # FASE 3 - detokenize sentence (do n't => don't)
                n = len(punt_w[0])
                if n >= 1:
                    idx = n-1
                    punt_w[0][idx] = punt_w[0][idx] + " "
                detk_w = [d.detokenize(m) for m in punt_w]

                # FASE 4 - remove special chars
                spch_w = [self.post_filter_char(m) for m in detk_w]
                final_list = spch_w[0].split(" ")
                num_lb = len(final_list)

                for w in final_list:
                    if w != "":
                        temp_text.append(w)
                        temp_labels.append(label)
                        
            preprocessed_text.append(temp_text)
            fixed_labels.append(temp_labels)

        return preprocessed_text, fixed_labels


    def save(self, path, corpus, labels):

        with open(path, 'w', encoding='utf-8') as f:

            for frase, label in zip(corpus, labels):
                for fr, lab in zip(frase, label):
                    if "\n" in fr:
                        # splitto e aggiungo le parole separate con la label spalmata
                        word_list = fr.split("\n")
                        for w in word_list:
                            f.write(w + " " + lab + "\n")
                    else:
                        # aggiungo parola, spazio, label
                        f.write(fr + " " + lab + "\n")
                f.write("\n") 
        f.close()