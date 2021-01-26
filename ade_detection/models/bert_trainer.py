#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW
from tqdm import tqdm
from os import path
import pandas as pd
import numpy as np
import random
import pickle
import torch
import gc
import os

from ade_detection.domain.train_config import TrainConfig
from ade_detection.models.bert_crf import Bert_CRF
from ade_detection.models.bert_lstm import Bert_LSTM
from ade_detection.models.bert_wrapper import Bert_wrapper
import ade_detection.utils.localizations as loc
from ade_detection.domain.enums import *
from ade_detection.domain.task import Task
from ade_detection.services.model_service import ModelService
import ade_detection.utils.file_manager as fm

class BertTrainer(object):


    def __init__(self, task: Task):
        self.task = task
        self.TEST_ONLY = task.train_mode == TRAIN_MODE.JUST_TESTING
        if task.train_mode == TRAIN_MODE.VALIDATION:
            self.train_dataset = task.split.to_tensor_dataset(task.split.train)
            self.validation_dataset = task.split.to_tensor_dataset(task.split.validation)
        elif task.train_mode == TRAIN_MODE.TESTING:
            self.train_dataset = task.split.to_tensor_dataset(task.split.train + task.split.validation)
            self.validation_dataset = task.split.to_tensor_dataset(task.split.test)
        elif task.train_mode == TRAIN_MODE.JUST_TESTING:
            self.train_dataset = None
            self.validation_dataset = task.split.to_tensor_dataset(task.split.test)

        LOG.info('cuda selection...')
        if torch.cuda.is_available():    
            self.device = torch.device('cuda')
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device('cpu')
        pd.set_option('precision', 2)

        LOG.info('init model in progress...')
        model_svc = ModelService(task)
        self.tokenizer = model_svc.get_tokenizer()
        self.config = model_svc.get_config()
        
        model_classes = {
            ARCHITECTURE.BERT_WRAPPER: Bert_wrapper,
            ARCHITECTURE.BERT_CRF: Bert_CRF,
            ARCHITECTURE.BERT_LSTM: Bert_LSTM,
        }
        
        MODEL_CLASS = model_classes[task.architecture]
        
        if self.TEST_ONLY:
            self.model = MODEL_CLASS.from_pretrained(task.model.value, config=self.config)
        else:
            self.model = MODEL_CLASS(self.config)
        
        self.model.to(self.device) # Runs the model on the GPU (if available)

        LOG.info('init completed successfully!')
        
        train_dataloader, validation_dataloader = self.make_dataloaders(self.train_dataset,
                                                                        self.validation_dataset)
        LOG.info('Dataloader fitted')
        
        val_df = self.init_val_df(self.validation_dataset)

        (val_df, df) = self.do_train_val(train_dataloader, validation_dataloader, val_df)
        LOG.info('Train/Val/Test completed successfully!')
        
        if not self.TEST_ONLY and task.train_mode != TRAIN_MODE.VALIDATION:
            LOG.info('model weights caching in progress...')
            if not path.exists(loc.abs_path([loc.TMP, loc.MODELS])):
                os.mkdir(loc.abs_path([loc.TMP, loc.MODELS]))
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(loc.abs_path([loc.TMP, loc.MODELS, self.task.id]))
            LOG.info('model weights saved successfully!')
            
        LOG.info('results export in progress...')

        try:
            val_df = self.add_detok_preds(val_df)
            LOG.info('detokenized successfully!')
        except:
            LOG.warning('detokenization issue!')
        
        self.task.val_df = val_df
        self.task.df = df
        fm.to_pickle(self.task, loc.abs_path([loc.TMP, self.task.id + '.pickle']))
        LOG.info('export completed successfully!')


    def init_val_df(self, validation_dataset):
        
        val_df_index = [int(d[-1]) for d in validation_dataset]
        
        if not self.TEST_ONLY:
            val_df_columns = ['numeric_id', 'input_ids', 'input_mask', 'gold_labels'] + \
                             [f'preds_{i+1}' for i in range(self.task.train_config.epochs)]
        else:
            val_df_columns = ['numeric_id', 'input_ids', 'input_mask', 'gold_labels'] + \
                             [f'preds_1']
            
        val_df = pd.DataFrame(index=val_df_index, columns=val_df_columns)
        
        for d in validation_dataset:
            val_df.at[int(d[3]), 'input_ids'] = d[0].tolist()   #   [0]: input ids 
            val_df.at[int(d[3]), 'input_mask'] = d[1].tolist()  #   [1]: attention masks
            val_df.at[int(d[3]), 'gold_labels'] = d[2].tolist() #   [2]: labels
            val_df.at[int(d[3]), 'numeric_id'] = int(d[3])      #   [3]: numeric id
            
        return val_df
        
    def make_dataloaders(self, train_dataset, validation_dataset):
        def _init_fn():
            np.random.seed(self.task.train_config.random_seed)
        if train_dataset is None:
            train_dataloader = None
        else:
            train_dataloader = DataLoader(
                train_dataset,                         
                sampler = RandomSampler(train_dataset),
                batch_size = self.config.batch_size,
                num_workers = 0,
                worker_init_fn=_init_fn )
        validation_dataloader = DataLoader(
            validation_dataset,                             
            sampler = SequentialSampler(validation_dataset),
            batch_size = self.config.batch_size,
            num_workers = 0,
            worker_init_fn=_init_fn )
        return train_dataloader, validation_dataloader
        
    def train_one_epoch(self, train_dataloader, optimizer, scheduler, epoch_i):
        # LOG.info(f'epoch {epoch_i} started')

        total_train_loss = 0
        self.model.train()

        for batch in train_dataloader:

            b_input_ids = batch[0].to(self.device)  #   [0]: input ids 
            b_input_mask = batch[1].to(self.device) #   [1]: attention masks
            b_labels = batch[2].to(self.device)     #   [2]: labels

            self.model.zero_grad()        

            (loss, _) = self.model( b_input_ids, 
                                    attention_mask = b_input_mask,
                                    labels = b_labels )

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        return avg_train_loss
        
        
    def do_train_val(self, train_dataloader, validation_dataloader, val_df):
        LOG.info('Train started')
        
        training_stats = []
        all_epoch_preds = []

        if not self.TEST_ONLY:
            
            optimizer = AdamW(self.model.parameters(), 
                              lr = self.task.train_config.learning_rate, 
                              eps = self.task.train_config.epsilon)
            total_steps = len(train_dataloader) * self.task.train_config.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps = 0,
                                                        num_training_steps = total_steps)
            
            for epoch_i in tqdm(range(self.task.train_config.epochs)):
                # training
                avg_train_loss = self.train_one_epoch(train_dataloader, optimizer, 
                                                      scheduler, epoch_i)
                # validation
                epoch_predictions, avg_val_loss = self.test_model(validation_dataloader, 
                                                                  val_df, epoch_i + 1)
                all_epoch_preds.append(epoch_predictions)

                training_stats.append({
                        'epoch': epoch_i + 1,
                        'Training_Loss': avg_train_loss,
                        'Valid_Loss': avg_val_loss,
                    })
            del optimizer
            del scheduler
        
        else:
            epoch_predictions, avg_val_loss = self.test_model(validation_dataloader, val_df, 1)
            all_epoch_preds.append(epoch_predictions)

            training_stats.append({
                    'epoch': 1,
                    'Training_Loss': -1,
                    'Valid_Loss': avg_val_loss,
                })
        

        LOG.info(f'Computing stats')    

        df_stats = pd.DataFrame(data=training_stats)  # training statistics
        df_stats = df_stats.set_index('epoch')        # Use the 'epoch' as the row index
        df_stats = df_stats[['Training_Loss','Valid_Loss']]
        LOG.info(df_stats)
        
        gc.collect()
        torch.cuda.empty_cache()

        return (val_df, df_stats)
    
    
    def test_model(self, validation_dataloader, val_df, epoch):

        epoch_predictions = []
        total_eval_loss = 0
        
        self.model.eval()

        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(self.device)   #   [0]: input ids 
            b_input_mask = batch[1].to(self.device)  #   [1]: attention masks
            b_labels = batch[2].to(self.device)      #   [2]: labels (BILUO)
            b_numeric_ids = batch[3].to(self.device) #   [3]: numeric id

            with torch.no_grad():
                (loss, preds) = self.model( b_input_ids, 
                                             attention_mask = b_input_mask,
                                             labels = b_labels )

            total_eval_loss += loss.item()

            for i, pred_tensor in enumerate(preds):
                pred = pred_tensor.tolist()
                val_df.at[int(b_numeric_ids[i]), f'preds_{epoch}'] = pred
                epoch_predictions.append([pred, int(b_numeric_ids[i])])

        # LOG.info(f'test competed, compute loss')

        # Average loss on all batches
        avg_val_loss = total_eval_loss / len(validation_dataloader)       
        return epoch_predictions, avg_val_loss


    def add_detok_preds(self, val_df):
        
        pred_cols = [c for c in val_df.columns if c.startswith('preds')]
        val_df['detok_gold_labels'] = ''
        val_df['detok_sent'] = ''
        val_df['detok_input_mask'] = ''
        
        for col in pred_cols:
            val_df['detok_'+col] = ''
        
        for _, line in val_df.iterrows():
            from_id_to_tok = self.tokenizer.convert_ids_to_tokens(line.input_ids)
            
            for col in pred_cols:
                detok_sent, detok_labels, detok_preds = self.mergeTokenAndPreserveData( from_id_to_tok,
                                                                                        line.gold_labels,
                                                                                        line[col] )
                line.at['detok_'+col] = detok_preds
            
            line['detok_gold_labels'] = detok_labels
            line['detok_sent'] = detok_sent
            line['detok_input_mask'] = [0 if x == self.tokenizer.pad_token else 1 for x in detok_sent]

        return val_df


    def mergeTokenAndPreserveData(self, sentence, labels, predictions):

        detok_sent = []
        detok_labels = []
        detok_predict = []

        for token, lab, pred in zip(sentence, labels, predictions):

            # CASE token to be added to the previous token
            if '##' in token:

                # rebuild the word
                detok_sent[-1] = detok_sent[-1] + token[2:]

                if pred > detok_predict[-1]:
                    detok_predict[-1] = pred
                #    LOG.info(' > Prediction updated')

            else:
                detok_sent.append(token)
                detok_labels.append(lab)
                detok_predict.append(pred)

        return detok_sent, detok_labels, detok_predict