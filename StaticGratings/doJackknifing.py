"""
Created on Fri Jul 20 15:22:06 2018
Get decoding contribution per neuron using jackknifing
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
from func_definePath import get_path
from func_Decoders import bayesian_decoding, lda_classification
from os.path import isfile, join

# Settings
num_splits = 5
group_size = 15
num_it = 500
phase = 0
spat_freq = 0.04
decoder = bayesian_decoding
# cre_line = 'Emx1-IRES-Cre'
cre_line = ['Cux2-CreERT2', 'Emx1-IRES-Cre']
area = ['VISp']
root_path = get_path()

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=area, cre_lines=cre_line)

# Loop through recordings
for i in range(0, len(ecs)):
# for i in range(0, 1):
    # Load in data
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.STATIC_GRATINGS])[0]
    print('Decoding recording ' + str(exp['id']) + ' [' + str(i+1) + ' of ' + str(len(ecs)) + ']')
    if isfile(join(root_path, 'boc/ophys_processed/', str(exp['id']) + '_neuron_contribution.npy')):
        print('Contribution file found, skipping..')
        pass
    else:        
        resp_mat = np.load(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
        stim_data = pd.read_pickle(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    
        # Select trials
        decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq) & (stim_data.phase == phase)])
        decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq) & (stim_data.phase == phase)]
    
        # Loop over neurons
        contr_mat = np.empty([num_it, len(resp_mat[0])])
        for n in range(len(resp_mat[0])):
            for j in range(num_it):
                # Get subsample and decode with and without neuron n
                neurons = np.append(n, np.random.choice(len(decode_resp[0]), group_size-1, replace=False))
                perf_all = decoder(decode_resp, decode_ori, neurons, num_splits)
                perf_excl = decoder(decode_resp, decode_ori, neurons[1:len(neurons)], num_splits)
                contr_mat[j,n] = group_size*perf_all-(group_size-1)*perf_excl
        contr = np.mean(contr_mat, axis=0)
        np.save(join(root_path, 'boc/ophys_processed/', str(exp['id']) + '_neuron_contribution'), contr)







