"""
Created on Sun Jul 22 13:08:17 2018
Perform Greedy decoding
@author: Guido Meijer
"""


from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
from func_Decoders import bayesian_decoding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Settings
num_splits = 5
temp_freq = 1
# cre_line = 'Emx1-IRES-Cre'
cre_line = 'Cux2-CreERT2'
area = 'VISp'

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=[area], cre_lines=[cre_line])

greedy_perf = []
for i in range(0, len(ecs)):
# for i in range(0, 1):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    resp_mat = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    neuron_contr = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_neuron_contribution.npy')
    
    # Select trials
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)]
    
    # Order neurons on decoding contribution
    sort_ind = np.argsort(neuron_contr)
    sort_mat = np.flip(decode_resp[:,sort_ind], 1)
    
    # Loop over neurons starting with the best
    perf = np.empty([len(sort_mat[0]), 1])
    for n in range(2, len(sort_mat[0])):
        neurons = np.array(range(n))
        perf[n] = bayesian_decoding(sort_mat, decode_ori, neurons, num_splits)
    greedy_perf.append([perf])

# Plot results
for i in range(len(greedy_perf)):
    this_perf = np.squeeze(np.array(greedy_perf[i]))
    plt.plot(range(2,len(this_perf)+2), this_perf)
plt.ylabel('Decoding performance')
plt.xlabel('Neurons sorted by best')

plt.savefig('/home/guido/Projects/AllenBrainDecoding/Plots/BayesianDecoding/GreedyDecoding_' + cre_line + '_' + area)
plt.show()
        
        
    
    
    