"""
Created on Sun Jul 22 14:44:43 2018
Correlate decoding contribution with other metrics as OSI
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
cre_line = 'Emx1-IRES-Cre'
area = 'VISp'

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=[area], cre_lines=[cre_line])

# Get all cells
cells = boc.get_cell_specimens()
cells = pd.DataFrame.from_records(cells)

greedy_perf = []
# for i in range(0, len(ecs)):
for i in range(0, 1):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    cells = boc.get_cell_specimens(experiment_container_ids=[ecs[i]['id']])
    cells = pd.DataFrame.from_records(cells)
    print("total cells: %d" % len(cells)) 
    
    
    
    
    

