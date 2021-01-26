#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from pandas.core.frame import DataFrame
from io import StringIO
from tqdm import tqdm
import pandas as pd 
import numpy as np
from os import path
import glob
import os
import re 

from ade_detection.services.database_service import DatabaseService
from ade_detection.importer.base_importer import BaseImporter
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.interval import Interval
from ade_detection.domain.span import Span
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class CadecImporter(BaseImporter):

    '''Importer script for the CADEC dataset (v2)
    see also https://data.csiro.au/collections/collection/CIcsiro:10948/SQcadec/RP1/RS25/RORELEVANCE/STsearch-by-keyword/RI1/RT1/
    '''

    def __init__(self):
        db = DatabaseService()
        
        self.decompress_dataset(loc.CADEC_ZIP_PATH, loc.CADEC_ARCHIVE_PATH)
        (corpus, annotations) = self.load_dataset()
        session = db.new_session()
        documents = self.encode_dataset(corpus, annotations)
        LOG.info('saving dataset...')
        session.add_all(documents)
        session.commit()
        LOG.info('dataset stored in the database successfully!')
        

    def encode_dataset(self, corpus, annotations):
        documents = []
        LOG.info('dataset serialization in progress...')
        for _, row in tqdm(corpus.iterrows(), total=len(corpus)):
            docs = list(filter(lambda x: x.external_id == row.text_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.text_id, 
                               text = row.text, 
                               corpus = CORPUS.CADEC)
                documents.append(doc)
            else: 
                doc = docs[0]

            for _, span in annotations[annotations.text_id == row.text_id].iterrows():
                if span.type == 'original':
                    chunks = re.split(' |;', span.raw_type)
                    intervals = []
                    for j in range(1, len(chunks) - 1, 2):
                        intervals.append(Interval(begin = int(chunks[j]), end = int(chunks[j+1])))
                    span_annotations = [Annotation(key = annotation_by_name(chunks[0]), value = span.span)]
                doc.spans.append(Span(intervals = intervals, annotations=span_annotations))

        return documents


    def load_dataset(self):
        LOG.info('dataset loading in progress...')
        filenames = glob.glob(loc.CADEC_TEXTS_QUERY)
        corpus = annotations = pd.DataFrame({})
        for filename in tqdm(filenames):
            text = self.load_text(filename)  

            if len(text['text'].loc[0]) == 0:
                continue
            corpus = pd.concat([corpus, text], axis=0).reset_index(drop=True)

            df = self.load_annotations(filename, text)
            if not df.empty:
                annotations = pd.concat([annotations, df], axis=0).reset_index(drop=True)
        LOG.info('dataset loaded successfully!...')

        return (corpus, annotations) 


    def load_annotations(self, text_path: str, text: str) -> DataFrame:  
        # paths
        meddra_path = text_path.replace(loc.TEXT, loc.MEDDRA).replace('.txt', '.ann')
        original_path = text_path.replace(loc.TEXT, loc.ORIGINAL).replace('.txt', '.ann')
        sct_path = text_path.replace(loc.TEXT, loc.SCT).replace('.txt', '.ann')

        # load annotations
        text_id = os.path.basename(original_path).replace('.ann', '') 
        df = self.read_csv(original_path, ['id', 'raw_type', 'span']) 
        original_annotations = df[~df['id'].str.contains('#')]
        original_annotations['type'] = 'original'
        original_annotations['text_id'] = text_id
        note_annotations =  df[df['id'].str.contains('#')] 
        note_annotations['type'] = 'note'
        note_annotations['text_id'] = text_id
        meddra_annotations = self.read_csv(meddra_path, ['id', 'raw_type', 'span']) 
        meddra_annotations['type'] = 'meddra'
        meddra_annotations['text_id'] = text_id
        sct_annotations = self.read_csv(sct_path, ['id', 'raw_type', 'span'])    
        sct_annotations['type'] = 'sct'
        sct_annotations['text_id'] = text_id
        
        return pd.concat([original_annotations, meddra_annotations, sct_annotations]).reset_index(drop=True)


    def read_csv(self, path, names):
        content = ''
        with open(path, 'r', errors='ignore') as file:
            content = file.read()
            for b in re.findall(r'[ ]{4,}', content):
                content = content.replace(b, '\t')

        return pd.read_csv(StringIO(content), sep='\t', header=None, 
                           names=names) 


    def load_text(self, text_path: str) -> DataFrame: 
        df = pd.DataFrame({'text_id': [''], 'text': ['']})
        text_id = os.path.basename(text_path).replace('.txt', '') 
        with open(text_path, 'r') as file:
            df.iloc[0] = [text_id, file.read().replace('\n', ' ')]
        return df