# -*- coding: utf-8 -*-
#=============================================================================
#
# Author: yuegang - 547258547@qq.com
# File created: 2018-09-27 16:40
# Last modified: 2018-09-28 15:57
# Filename: hmm.py
# Description: 
#
#=============================================================================
import pickle
import time
import os
import numpy as np
import pandas as pd
from copy import deepcopy

train_path = 'data/train.in'
test_path = 'data/test.in'
output_path = 'data/test.out'


def train(train_path):
    print('start training')
    df = pd.read_csv(train_path, '\t', names=['word', 'label'])
    O_arr = np.array(list(set(df['word'])))
    I_arr = np.array(list(set(df['label']))).astype('str')
    O_dict = {word: idx for idx, word in enumerate(O_arr)}
    I_dict = {label: idx for idx, label in enumerate(I_arr)}

    A_arr = np.zeros((len(I_arr), len(I_arr)))
    B_arr = np.zeros((len(I_arr), len(O_arr)))
    PI_arr = np.zeros(len(I_arr))
    first_state = True
    pre_i_idx = None

    fp = open(train_path, 'r')
    for line in fp:
        segs = line.strip().split()
        if len(segs) != 2:
            first_state = True
            pre_i_idx = None
            continue
        o = segs[0]
        i = segs[1]
        o_idx = O_dict[o]
        i_idx = I_dict[i]
        B_arr[i_idx][o_idx] += 1
        if first_state:
            first_state = False
            PI_arr[i_idx] += 1
        else:
            A_arr[pre_i_idx][i_idx] += 1
        pre_i_idx = i_idx
    fp.close()

    A_arr = ((A_arr.T + 1.) / (np.sum(A_arr, axis=1) + 1.)).T
    B_arr = ((B_arr.T + 1.) / (np.sum(B_arr, axis=1) + 1.)).T
    PI_arr = (PI_arr + 1.) / (np.sum(PI_arr) + 1.)

    print('finished training')
    return O_arr, I_arr, A_arr, B_arr, PI_arr


def save_model(path, O_arr, I_arr, A_arr, B_arr, PI_arr):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'O_arr'), O_arr)
    np.save(os.path.join(path, 'I_arr'), I_arr)
    np.save(os.path.join(path, 'A_arr'), A_arr)
    np.save(os.path.join(path, 'B_arr'), B_arr)
    np.save(os.path.join(path, 'PI_arr'), PI_arr)


def load_model(path):
    O_arr = np.load(os.path.join(path, 'O_arr.npy'))
    I_arr = np.load(os.path.join(path, 'I_arr.npy'))
    A_arr = np.load(os.path.join(path, 'A_arr.npy'))
    B_arr = np.load(os.path.join(path, 'B_arr.npy'))
    PI_arr = np.load(os.path.join(path, 'PI_arr.npy'))
    return O_arr, I_arr, A_arr, B_arr, PI_arr


def predict(X, O_arr, O_dict, I_arr, A_arr, B_arr, PI_arr):
    path = []
    cur_prob = deepcopy(PI_arr)
    for idx, x in enumerate(X):
        if idx != 0:
            cur_prob = np.dot(cur_prob, A_arr)
        i_idx = O_dict.get(x, -1)
        if i_idx != -1:
            cur_prob *= B_arr.T[i_idx]
        cur_i_idx = np.argmax(cur_prob)
        path.append(I_arr[cur_i_idx])
    return np.array(path)


def test(O, I, A, B, PI):
    print('start testing')
    O_dict = {word: idx for idx, word in enumerate(O)}
    with open(test_path, 'r') as infile, \
         open(output_path, 'w') as outfile:
        X_test = []
        y_test = []
        for line in infile:
            segs = line.strip().split('\t')
            if len(segs) != 2:
                if len(X_test) == 0:
                    continue
                preds = predict(X_test, O, O_dict, I, A, B, PI)
                for vals in zip(X_test, y_test, preds):
                    outfile.write('\t'.join(vals) + '\n')
                outfile.write('\n')
                X_test = []
                y_test = []
            else:
                o = segs[0]
                s = segs[1]
                X_test.append(o)
                y_test.append(s)
    print('finished testing')


if __name__ == '__main__':
    time1 = time.time()
    O_arr, I_arr, A_arr, B_arr, PI_arr = train(train_path)
    save_model('model', O_arr, I_arr, A_arr, B_arr, PI_arr)
    O_arr, I_arr, A_arr, B_arr, PI_arr = load_model('model')
    time2 = time.time()
    print('Train Time: {}'.format(time2 - time1))
    test(O_arr, I_arr, A_arr, B_arr, PI_arr)
    time3 = time.time()
    print('Test Time: {}'.format(time3 - time2))
    print('All Time: {}'.format(time3 - time1))
