#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:50:36 2018

Plot decoding of different cre lines

@author: Guido Meijer
"""

from os import listdir
from os.path import join
from func_definePath import get_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import func_General as fg

root_path = get_path()
file_list = listdir(join(root_path, 'boc/ophys_decoding_results/'))

for i in file_list:
    decode_perf = pd.read_pickle(join(root_path, 'boc/ophys_decoding_results/', i))
    line = plt.plot(list(decode_perf.index.values), np.mean(decode_perf, axis=1), label=i[17:21])
    plt.fill_between(list(decode_perf.index.values), np.mean(decode_perf, axis=1)-fg.stderr(decode_perf, axis=1), \
                     np.mean(decode_perf, axis=1)+fg.stderr(decode_perf, axis=1), alpha = 0.3)

plt.ylabel('Decoding performance')
plt.xlabel('Neurons')
plt.legend()
plt.savefig(join(root_path, 'Plots/BayesianDecoding/DecodingCreLines'))
plt.show()
