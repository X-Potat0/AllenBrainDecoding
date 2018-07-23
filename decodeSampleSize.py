"""
Created on Fri Jul 20 15:22:06 2018
Perform decoding of orientation using different sizes of random subsets
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
from func_Decoders import bayes_decoding

# Settings
num_splits = 5
group_sizes = range(5,5,160)
num_it = 500
temp_freq = 1
cre_line = 'Emx1-IRES-Cre'
area = 'VISp'

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=[area], cre_lines=[cre_line])

all_perf = []
# for i in range(0, len(ecs)):
for i in range(0, 1):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    resp_mat = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    
    # Select trials
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)]
    
    for n in group_sizes:
        print('Group size ' + n)
        for i in range(num_it):
            neurons = 
            bayes_decoding(decode_resp, decode_ori, neurons, )
        
    
    




