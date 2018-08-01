"""
Created on Fri Jul 20 15:27:04 2018
Set of functions to perform decoding
@author: Guido Meijer
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np

def bayesian_decoding(resp, stim, neurons, num_splits):
    kf = KFold(n_splits=num_splits)
    gnb = GaussianNB()
    perf = 0
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        gnb.fit(train_resp[:, neurons], stim[train_index])
        y_pred = gnb.predict(test_resp[:, neurons])
        perf = perf + np.sum(stim[test_index] == y_pred) / len(y_pred)
    perf = perf / num_splits
    return perf
