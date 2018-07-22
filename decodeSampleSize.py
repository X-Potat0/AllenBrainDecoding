"""
Created on Fri Jul 20 15:22:06 2018
Perform decoding of orientation using different sizes of random subsets
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd

# Settings
num_splits = 5
group_sizes = range(5,5,160)
num_it = 500
temp_freq = 1
cre_line = 'Emx1-IRES-Cre'
area = 'VISp'

for i in 



