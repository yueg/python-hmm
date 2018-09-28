# -*- coding: utf-8 -*-
#=============================================================================
#
# Author: gangyue - gangyue@iflytek.com
# File created: 2018-09-27 10:57
# Last modified: 2018-09-27 10:57
# Filename: evaluate.py
# Description: 
#
#=============================================================================
from sklearn.metrics import classification_report

label_dict = {
    'S': 0,
    'B': 1,
    'M': 2,
    'E': 3
}

y_true = []
y_pred = []
with open('data/test.out', 'r') as fp:
    for line in fp:
        line = line.strip()
        if len(line.split()) != 3:
            continue
        rs = line.split()[1:]
        y_true.append(label_dict[rs[0]])
        y_pred.append(label_dict[rs[1]])
target_names = [k for k, _ in label_dict.items()]
print(classification_report(y_true, y_pred, target_names=target_names))
