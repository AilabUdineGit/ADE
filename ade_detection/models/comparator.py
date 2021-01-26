#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import os
import pandas as pd
from copy import deepcopy

from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from ade_detection.services.model_service import ModelService
from ade_detection.models.evaluator import Evaluator
from ade_detection.utils.metrics import Metrics
import ade_detection.utils.localizations as loc 
import ade_detection.utils.file_manager as fm
from ade_detection.utils.graphics import *
from ade_detection.domain.enums import *
from ade_detection.domain.task import Task
from ade_detection.domain.best import Best


class Comparator(object):

    def __init__(self, tasks:list, result_filename:str, detok = True, model_filename = None, just_last=False):
        self.tasks = tasks        
        self.DETOK = 'detok_' if detok else ''
        self._DETOK = '_detok' if detok else ''

        for task in tasks:
            model_service = ModelService(task)
            self.tokenizer = model_service.get_tokenizer()
            task = self.getAllMetrics(task)
            fm.to_pickle(task, loc.abs_path([loc.TMP, task.id+".pickle"]))
        
        if just_last:
            best_partial = self.find_last_configuration(tasks, COMPARATOR_MODE.PARTIAL)
            best_strict = self.find_last_configuration(tasks, COMPARATOR_MODE.STRICT)   
        else:
            best_partial = self.find_best_configuration(tasks, COMPARATOR_MODE.PARTIAL)
            best_strict = self.find_best_configuration(tasks, COMPARATOR_MODE.STRICT)    
        
        print(f"\n\nBEST PARTIAL:\n\n{self.best_to_str(best_partial)}")
        print(f"\n\nBEST STRICT:\n\n{self.best_to_str(best_strict)}")
        best = Best(best_strict, best_partial)
        fm.to_pickle(best, loc.abs_path([loc.TMP, result_filename]))


    def convert_ids_tags_to_iob(self, tags, task):
        iob_tags = []
        for i, tag in enumerate(tags):
            if tag <= 0:
                if tag == 0:
                    iob_tags.append('O')
            else:
                annotation_index = int((tag + 1)/2) - 1
                if tag % 2 == 0:
                    iob_tags.append('B-'+task.goal[annotation_index].name)
                else: 
                    iob_tags.append('I-'+task.goal[annotation_index].name)
        return iob_tags


    def getAllMetrics(self, task):
        LOG.info('\n\n\n' + task.id + '\n\n\n')

        df = task.val_df
        loss_df = task.df

        tr_loss = loss_df['Training_Loss']
        vl_loss = loss_df['Valid_Loss']

        to_keep = df[f'{self.DETOK}input_mask'].tolist()
        to_keep = [t.count(1) for t in to_keep]
        
        all_true = df[f'{self.DETOK}gold_labels'].tolist()
        all_true = [self.convert_ids_tags_to_iob(t, task) for t in all_true]
        all_true = [t[1:to_keep[i]] for i,t in enumerate(all_true)]

        # dizionario contenente i dati di tutte le epoche
        results_dict = {}
        
        for e in range(task.train_config.epochs):
            all_pred = df[f'{self.DETOK}preds_{e+1}'].tolist()
            all_pred = [self.convert_ids_tags_to_iob(t, task) for t in all_pred]
            all_pred = [t[1:to_keep[i]] for i,t in enumerate(all_pred)]
            
            evaluator = Evaluator(all_true, all_pred, [x.name for x in task.goal])
            
            results, results_agg = evaluator.evaluate()
            results_dict[e] = {'results':results, 'results_agg':results_agg}

        str_correct = []
        str_incorrect = []
        str_partial = []
        str_missed = []
        str_spurious = []
        str_possible = []
        str_actual = []
        str_precision = []
        str_recall = []

        prt_correct = []
        prt_incorrect = []
        prt_partial = []
        prt_missed = []
        prt_spurious = []
        prt_possible = []
        prt_actual = []
        prt_precision = []
        prt_recall = []

        for key, value in results_dict.items():
            str_correct.append(value['results']['strict']['correct'])
            str_incorrect.append(value['results']['strict']['incorrect'])
            str_partial.append(value['results']['strict']['partial'])
            str_missed.append(value['results']['strict']['missed'])
            str_spurious.append(value['results']['strict']['spurious'])
            str_possible.append(value['results']['strict']['possible'])
            str_actual.append(value['results']['strict']['actual'])
            str_precision.append(value['results']['strict']['precision'])
            str_recall.append(value['results']['strict']['recall'])

            prt_correct.append(value['results']['partial']['correct'])
            prt_incorrect.append(value['results']['partial']['incorrect'])
            prt_partial.append(value['results']['partial']['partial'])
            prt_missed.append(value['results']['partial']['missed'])
            prt_spurious.append(value['results']['partial']['spurious'])
            prt_possible.append(value['results']['partial']['possible'])
            prt_actual.append(value['results']['partial']['actual'])
            prt_precision.append(value['results']['partial']['precision'])
            prt_recall.append(value['results']['partial']['recall'])

        
        list_of_names = [f'{self.DETOK}preds_{e+1}' for e in range(task.train_config.epochs)]

        metrics_strict_df = pd.DataFrame(
            {'epoch': list_of_names,
            'str_correct': str_correct,
            'str_incorrect': str_incorrect,
            'str_partial': str_partial,
            'str_missed': str_missed,
            'str_spurious': str_spurious,
            'str_possible': str_possible,
            'str_actual': str_actual,
            'str_precision': str_precision,
            'str_recall': str_recall,
            'training_loss': tr_loss,
            'validation_loss' : vl_loss
            })

        metrics_partial_df = pd.DataFrame(
            {'epoch': list_of_names,
            'prt_correct': prt_correct,
            'prt_incorrect': prt_incorrect,
            'prt_partial': prt_partial,
            'prt_missed': prt_missed,
            'prt_spurious': prt_spurious,
            'prt_possible': prt_possible,
            'prt_actual': prt_actual,
            'prt_precision': prt_precision,
            'prt_recall': prt_recall,
            'training_loss': tr_loss,
            'validation_loss' : vl_loss
            })

        metrics_strict_df['str_f1score'] = metrics_strict_df.apply(lambda row: Metrics.get_f1score(row.str_precision,row.str_recall), axis=1)
        metrics_partial_df['prt_f1score'] = metrics_partial_df.apply(lambda row: Metrics.get_f1score(row.prt_precision,row.prt_recall), axis=1)
        task.metrics_strict_df = metrics_strict_df
        task.metrics_partial_df = metrics_partial_df
        return task
    
    
    def find_last_configuration(self, tasks:list, mode: COMPARATOR_MODE):
        best = None

        if len(tasks) > 1:
            LOG.info("This shouldn't happen in testing mode")
        
        task = tasks[0]
        
        
        df = task.metrics_strict_df if mode == COMPARATOR_MODE.STRICT else task.metrics_partial_df
        df.columns = [x.replace("prt_", "").replace("str_", "") for x in df.columns]
        
        line = df.iloc[-1]

        tr_loss = line.training_loss
        vl_loss = line.validation_loss
        f1 = line.f1score
        precision = line.precision
        recall = line.recall 
        epoch = int(line.epoch.split("_")[-1])

        curr_f1 = f1
        curr_precision = precision
        curr_recall = recall
        curr_epoch = epoch
        curr_val_loss = vl_loss
        curr_task = task
            
        best = deepcopy(curr_task)
        best.best_f1 = curr_f1
        best.precision = curr_precision
        best.recall = curr_recall
        best.best_val_loss = curr_val_loss 
        best.epochs = curr_epoch
                
        return best
        


    def find_best_configuration(self, tasks:list, mode: COMPARATOR_MODE):
        best = None

        for task in tasks:
            df = task.metrics_strict_df if mode == COMPARATOR_MODE.STRICT else task.metrics_partial_df
            df.columns = [x.replace("prt_", "").replace("str_", "") for x in df.columns]
            curr_val_loss = 10000
            curr_epoch = -1
            curr_f1 = -1
            curr_precision = -1
            curr_recall = -1
            curr_task = task
                
            for _, line in df.iterrows():
                    
                tr_loss = line.training_loss
                vl_loss = line.validation_loss
                f1 = line.f1score
                precision = line.precision
                recall = line.recall 
                epoch = int(line.epoch.split("_")[-1])
                    
                if Metrics.overfit(tr_loss, vl_loss):
                    break

                if (f1 > curr_f1 and not Metrics.overfit(tr_loss, vl_loss)) or epoch == 1:
                    curr_f1 = f1
                    curr_precision = precision
                    curr_recall = recall
                    curr_epoch = epoch
                    curr_val_loss = vl_loss
                    curr_task = task
            
            if best is None or best.best_f1 < curr_f1:
                
                best = deepcopy(curr_task)
                best.best_f1 = curr_f1
                best.precision = curr_precision
                best.recall = curr_recall
                best.best_val_loss = curr_val_loss 
                best.epochs = curr_epoch
                
        return best
                

    def best_to_str(self, best):
        
        best_str = f'''
    F1 {best.best_f1}
    VAL_LOS {best.best_val_loss}
    EPOCH {best.epochs}
    PRECISION {best.precision}
    RECALL {best.recall}
    CONFIGS {best.id} '''
        
        best_str = f'''config_id\tlr\tdropout\tepoch\tarchitecture\tf1\tprecision\trecall
{best.id}\t{best.train_config.learning_rate}\t{best.train_config.dropout}\t{best.epochs}\t{best.model.name}{"+CRF" if "CRF" in best.architecture.name else ""}\t{best.best_f1}\t{best.precision}\t{best.recall}'''
        
        return best_str


    @staticmethod
    def compare_dirkson(corpus_name: str):
        df = pd.read_pickle(loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, 
                                          corpus_name + loc.DIRKSON_TEST_RESULTS_PICKLE]))
        all_true_array = df.labels.values
        all_pred_array = df.pred_labels.values

        all_true = []
        for t in all_true_array:
            all_true.append(t)
        all_pred = []
        for t in all_pred_array:
            all_pred.append(t)

        evaluator = Evaluator(all_true, all_pred, [""])
        results, results_agg = evaluator.evaluate()
        fm.to_json(results, loc.abs_path([loc.ASSETS, loc.MODELS, loc.DIRKSON, corpus_name + "-results.json"]))