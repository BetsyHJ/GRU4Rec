# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
#import gru4rec_source as gru4rec
import gru4rec_gitSource as gru4rec
import evaluation

PATH_TO_TRAIN = '../data/movie/train.oneout.csv'#'rsc15_train_full.txt'
#PATH_TO_TEST = '../data/test.oneout.csv'#'/path/to/rsc15_test.txt'

if __name__ == '__main__':
    print "the command is", " ".join(sys.argv)
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    '''
    print('Training GRU4Rec with 100 hidden units')    
    
    gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    '''
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    
    print('Training GRU4Rec with 100 hidden units')

    gru = gru4rec.GRU4Rec(loss='bpr', final_act='linear', hidden_act='tanh', layers=[256], batch_size=200, dropout_p_hidden=0.2, learning_rate=0.001, momentum=0.5, n_sample=10, sample_alpha=0, time_sort=True, n_epochs=10)
    gru.fit(data)
    gru.save_ItemEmbedding(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
