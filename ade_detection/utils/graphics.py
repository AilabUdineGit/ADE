#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = None
__version__ = '1.0'
__license__ = None
__copyright__ = None


import plotly.express as px
import plotly.graph_objects as go

import ade_detection.utils.localizations as loc


def make_graphics(metrics_strict_df, metrics_partial_df, TYPE, NOTATION, LR, DROPOUT, prefix, asset_folder):
    fig = go.Figure()

    # Create and style traces
    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_correct'], 
                            name='Correct',
                            line=dict(color='#06D6A0', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_incorrect'], 
                            name='Incorrect',
                            line=dict(color='#ef476f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_partial'], 
                            name='Partial',
                            line=dict(color='#118AB2', width=2)))
                        
    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_missed'], 
                            name='Missed',
                            line=dict(color='#f78c6b', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_spurious'], 
                            name='Spurious',
                            line=dict(color='#FFD166', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_possible'], 
                            name='Possible',
                            line=dict(color='#073B4C', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_actual'], 
                            name='Actual',
                            line=dict(color='mediumorchid', width=2)))

    # Edit the layout
    fig.update_layout(title='Strict metrics - ' + TYPE + ', ' + NOTATION + ', lr: ' + LR + ', dropout: ' + DROPOUT,
                    xaxis_title='Epochs',
                    yaxis_title='Value')

    #fig.show()
    fig.write_html(loc.abs_path([loc.ASSETS, asset_folder, f'{prefix}___strict___{TYPE}.html']))

    fig = go.Figure()

    # Create and style traces
    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_precision'], 
                            name='Precision',
                            line=dict(color='#f4a261', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_recall'], 
                            name='Recall',
                            line=dict(color='#e76f51', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['str_f1score'], 
                            name='F1-Score',
                            line=dict(color='#2a9d8f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['training_loss'], 
                            name='Training Loss',
                            line=dict(color='#ef476f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['validation_loss'], 
                            name='Validation Loss',
                            line=dict(color='mediumorchid', width=2)))
                        
    # Edit the layout
    fig.update_layout(title='Strict metrics - ' + TYPE + ', ' + NOTATION + ', lr: ' + LR + ', dropout: ' + DROPOUT,
                    xaxis_title='Epochs',
                    yaxis_title='Value')

    #fig.show()
    fig.write_html(loc.abs_path([loc.ASSETS, asset_folder, f'{prefix}___strict_prec_rec_f1___{TYPE}.html']))

    fig = go.Figure()

    # Create and style traces
    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_correct'], 
                            name='Correct',
                            line=dict(color='#06D6A0', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_incorrect'], 
                            name='Incorrect',
                            line=dict(color='#ef476f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_partial'], 
                            name='Partial',
                            line=dict(color='#118AB2', width=2)))
                        
    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_missed'], 
                            name='Missed',
                            line=dict(color='#f78c6b', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_spurious'], 
                            name='Spurious',
                            line=dict(color='#FFD166', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_possible'], 
                            name='Possible',
                            line=dict(color='#073B4C', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_actual'], 
                            name='Actual',
                            line=dict(color='mediumorchid', width=2)))

    # Edit the layout
    fig.update_layout(title='Partial metrics - ' + TYPE + ', ' + NOTATION + ', lr: ' + LR + ', dropout: ' + DROPOUT,
                    xaxis_title='Epochs',
                    yaxis_title='Value')

    #fig.show()
    fig.write_html(loc.abs_path([loc.ASSETS, asset_folder, f'{prefix}___partial___{TYPE}.html']))

    fig = go.Figure()

    # Create and style traces
    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_precision'], 
                            name='Precision',
                            line=dict(color='#f4a261', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_recall'], 
                            name='Recall',
                            line=dict(color='#e76f51', width=2)))

    fig.add_trace(go.Scatter(x=metrics_partial_df['epoch'], 
                            y=metrics_partial_df['prt_f1score'], 
                            name='F1-Score',
                            line=dict(color='#2a9d8f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['training_loss'], 
                            name='Training Loss',
                            line=dict(color='#ef476f', width=2)))

    fig.add_trace(go.Scatter(x=metrics_strict_df['epoch'], 
                            y=metrics_strict_df['validation_loss'], 
                            name='Validation Loss',
                            line=dict(color='mediumorchid', width=2)))
                        
    # Edit the layout
    fig.update_layout(title='Partial metrics - ' + TYPE + ', ' + NOTATION + ', lr: ' + LR + ', dropout: ' + DROPOUT,
                    xaxis_title='Epochs',
                    yaxis_title='Value')
    #fig.show()
    fig.write_html(loc.abs_path([loc.ASSETS, asset_folder, f'{prefix}___partial_prec_rec_f1___{TYPE}.html']))
