# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np, time
import pandas as pd
import gru4rec
import evaluation

PATH_TO_TRAIN = '../data/movie/train.oneout.csv'#'rsc15_train_full.txt'
#PATH_TO_TEST = '../data/test.oneout.csv'#'/path/to/rsc15_test.txt'

File_ItemEmbedding = sys.argv[1]
def read_ItemEmbedding(File_ItemEmbedding):
    ItemEmbedding = {}
    f = open(File_ItemEmbedding)
    #length = int(f.readline().strip().split()[1])
    for line in f.readlines():
        ss = line.strip().split()
	#if ss[0][0] != 'm':
	#    continue
        ItemId = int(ss[0][:])
        t = []
        for i in range(1, len(ss)):
            t.append(float(ss[i]))
        ItemEmbedding[ItemId] = np.array(t, dtype = np.float32)
    f.close()
    length = ItemEmbedding[ItemEmbedding.keys()[0]].shape[0]
    return ItemEmbedding, length

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})

    ItemEmbedding, length = read_ItemEmbedding(File_ItemEmbedding)
    print length, len(ItemEmbedding)
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
    start_time = time.time()
    gru = gru4rec.GRU4Rec(loss='bpr', final_act='linear', hidden_act='tanh', layers=[256], batch_size=200, embedding = length, dropout_p_hidden=0.2, n_sample=10, learning_rate=0.001, momentum=0.1, sample_alpha=0, time_sort=True, n_epochs=10, train_random_order=True)
    gru.fit(data, ItemEmbedding)
    ItemFile = 'item_embedding'
    if len(sys.argv)>2:
	ItemFile = sys.argv[2]
    gru.save_ItemEmbedding(data, ItemFile)#'item.embedding')

    UserFile = 'user.embedding'
    if len(sys.argv)>3:
	UserFile = sys.argv[3]
    evaluation.evaluate_sessions_batch(gru, valid, None, SaveUserFile = UserFile)#'user.embedding')
    end_time = time.time()
    print start_time, end_time
    print (start_time - end_time)
    #print('Recall@20: {}'.format(res[0]))
    #print('MRR@20: {}'.format(res[1]))
