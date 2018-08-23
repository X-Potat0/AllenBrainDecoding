"""
Created on Fri Jul 20 15:27:04 2018
Set of functions to perform decoding
@author: Guido Meijer
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import warnings

warnings.filterwarnings("ignore")

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

def lda_classification(resp, stim, neurons, num_splits):
    #clf = LinearDiscriminantAnalysis(solver='lsqr', priors=[len(np.unique(stim))]*len(np.unique(stim)))
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    kf = KFold(n_splits=num_splits)
    perf = 0
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp[:, neurons], stim[train_index])
        y_pred = clf.predict(test_resp[:, neurons])
        perf = perf + np.sum(stim[test_index] == y_pred) / len(y_pred)
    perf = perf / num_splits
    return perf

def lda_two_class(resp, stim, neurons, num_splits):
    clf = LinearDiscriminantAnalysis()
    kf = KFold(n_splits=num_splits)
    unique_stim = np.unique(stim)
    ori_perf = []
    for s in unique_stim:
        next_s = s+np.diff(unique_stim)[0]
        if next_s == 360:
            next_s = 0
        this_resp = resp[(stim == s) | (stim == next_s)]
        this_stim = stim[(stim == s) | (stim == next_s)]
        this_perf = 0
        for train_index, test_index in kf.split(this_resp):
            train_resp = this_resp[train_index]
            test_resp = this_resp[test_index]
            clf.fit(train_resp[:, neurons], this_stim[train_index])
            y_pred = clf.predict(test_resp[:, neurons])
            this_perf = this_perf + (np.sum(this_stim[test_index] == y_pred) / len(y_pred))
        ori_perf = np.append(ori_perf, this_perf/num_splits)
    perf = np.mean(ori_perf)
    return perf
        