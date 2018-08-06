"""
Created on Mon Aug  6 15:58:12 2018
Decode with best and worst neurons on shuffled and non-shuffled data
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
from func_Decoders import bayesian_decoding
import matplotlib.pyplot as plt
from func_definePath import get_path

# Settings
num_neurons = 15
num_splits = 5
temp_freq = 1
cre_line = ['Emx1-IRES-Cre', 'Cux2-CreERT2']
area = ['VISp']
root_path = get_path()

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=area, cre_lines=cre_line)

decode_df = pd.DataFrame()
# for i in range(0, len(ecs)):
for i in range(0, 1):
    # Load in data
    print('Processing recording ' + str(i+1) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    resp_mat = np.load(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    neuron_contr = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_neuron_contribution.npy')
    
    # Create dataframe with neural responses
    resp_df = pd.DataFrame(data=resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)], \
                 index=np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)]), \
                 columns=neuron_contr.argsort())
    resp_df = resp_df.reindex(sorted(resp_df.columns), axis=1)
    
    # Shuffle neural responses per orientation 
    shuffle_df = resp_df
    for n in list(resp_df):
        for ori in np.unique(list(resp_df.index.values)):
            shuffle_df.loc[ori, n] = shuffle_df.loc[ori, n].sample(frac=1)
            
            
            