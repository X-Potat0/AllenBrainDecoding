"""
Created on Fri Jul 20 15:22:06 2018
Perform decoding of orientation using different sizes of random subsets
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
from func_Decoders import bayesian_decoding, lda_classification, lda_two_class
import matplotlib.pyplot as plt
from func_definePath import get_path

# Settings
num_splits = 5
num_it = 500
phase = 0
spat_freq = 0.04
decoder = lda_classification
str_decoder = 'lda_classification'
cre_line = ['Emx1-IRES-Cre', 'Cux2-CreERT2']
# cre_line = ['Cux2-CreERT2']
area = ['VISp']
root_path = get_path()

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=area, cre_lines=cre_line)

decode_perf = []
for i in range(0, len(ecs)):
# for i in range(15, 16):
    # Load in data
    print('Decoding recording ' + str(i+1) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.STATIC_GRATINGS])[0]
    resp_mat = np.load(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
        
    # Select trials   
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq)]
    # decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.phase == phase) & (stim_data.spatial_frequency == spat_freq)])
    # decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.phase == phase) & (stim_data.spatial_frequency == spat_freq)]
    
    #Select neuronal group sizes
    group_sizes = range(10, len(decode_resp[0]), 10)
    
    all_perf = pd.DataFrame()
    for n in group_sizes:
        print('Group size ' + str(n))
        perf = np.array([])
        for j in range(num_it):
            neurons = np.random.choice(len(decode_resp[0]), n, replace=False)
            this_perf = decoder(decode_resp, decode_ori, neurons, num_splits)
            perf = np.append(perf, this_perf)
        all_perf[str(n)] = perf
    decode_perf.append(all_perf)
   
# np.save(root_path + 'boc/ophys_decoding_results/LDADecodingAllNeurons_' + area[0], decode_perf)

# Plot output
for i in range(len(decode_perf)):
    plt.plot(np.mean(decode_perf[i], axis=0))
plt.ylabel('Decoding performance')
plt.xlabel('Neurons')
plt.xticks(range(0, 41, 4))
plt.savefig('/home/guido/Projects/AllenBrainDecoding/Plots/StaticGratings/' + str_decoder + '/SampleSizeDecodingAllNeurons_' + area[0])
# plt.savefig(root_path + 'Plots/BayesianDecoding/SampleSizeDecoding_' + cre_line[0] + '_' + area[0])
plt.show()
    



        
        
    
    




