"""
Created on Mon Aug  6 15:58:12 2018
Decode with best and worst neurons on shuffled and non-shuffled data
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
from func_Decoders import bayesian_decoding, lda_classification, lda_two_class
import matplotlib.pyplot as plt
from func_definePath import get_path
import func_General as fg

# Settings
decoder = bayesian_decoding
str_decoder = 'bayesian_decoding'
num_neurons = 15
num_splits = 5
spat_freq = 0.04
phase = 0
cre_line = ['Emx1-IRES-Cre', 'Cux2-CreERT2']
area = ['VISp']
root_path = get_path()

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=area, cre_lines=cre_line)

# Initialize dataframe
decode_df = pd.DataFrame(columns={'best','middle','worst'})
decode_df = decode_df[['best','middle','worst']]
decode_shuf_df = pd.DataFrame(columns={'best','middle','worst'})
decode_shuf_df = decode_df[['best','middle','worst']]

for i in range(0, len(ecs)):
# for i in range(0, 3):
    # Load in data
    print('Processing recording ' + str(i+1) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.STATIC_GRATINGS])[0]
    resp_mat = np.load(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    neuron_contr = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_neuron_contribution.npy')
    
    # Create response matrix and stim index
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq) & (stim_data.phase == phase)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.spatial_frequency == spat_freq) & (stim_data.phase == phase)]
   
    # Sort matrix
    sort_ind = np.argsort(-neuron_contr) 
    sort_mat = decode_resp[:,sort_ind]
    
    # Perform decoding with n best, n worst, and n middle neurons
    best = np.arange(0,num_neurons)
    worst = np.arange(len(sort_mat[0])-num_neurons,len(sort_mat[0]))
    middle = best+int(len(sort_mat[0])/2)
    
    perf = decoder(sort_mat, decode_ori, best, num_splits)
    decode_df.loc[i,'best'] = perf
    perf = decoder(sort_mat, decode_ori, worst, num_splits)
    decode_df.loc[i,'worst'] = perf
    perf = decoder(sort_mat, decode_ori, middle, num_splits)
    decode_df.loc[i,'middle'] = perf
    
    # Shuffle neural responses per orientation 
    shuffle_df = pd.DataFrame(data=sort_mat, index=decode_ori, dtype = np.float64)
    for n in list(shuffle_df):
        for ori in np.unique(list(shuffle_df.index.values)):
            shuffle_df.loc[ori, n] = shuffle_df.loc[ori, n].sample(frac=1)
    shuffle_mat = shuffle_df.values
    
    # Perform decoding on shuffled dataset with n best, n worst, and n middle neurons
    perf = decoder(shuffle_mat, decode_ori, best, num_splits)
    decode_shuf_df.loc[i,'best'] = perf
    perf = decoder(shuffle_mat, decode_ori, worst, num_splits)
    decode_shuf_df.loc[i,'worst'] = perf
    perf = decoder(shuffle_mat, decode_ori, middle, num_splits)
    decode_shuf_df.loc[i,'middle'] = perf

# Plot output    
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(np.arange(len(list(decode_df))), np.mean(decode_df), yerr=fg.stderr(decode_df))
ax1.set_xticks(np.arange(len(list(decode_df))))
ax1.set_xticklabels(('Best', 'Middle', 'Worst'))
ax1.set_ylabel('Decoding performance')
ax1.set_title(str_decoder)

decode_diff = decode_shuf_df-decode_df
ax2.bar(np.arange(len(list(decode_diff))), np.mean(decode_diff), yerr=fg.stderr(decode_diff))
ax2.set_xticks(np.arange(len(list(decode_df))))
ax2.set_xticklabels(('Best', 'Middle', 'Worst'))
ax2.set_ylabel('Decoding increase over shuffled')

plt.tight_layout()
plt.savefig(root_path + 'Plots/StaticGratings/' + str_decoder + '/DecodingShuffled_' + area[0])
plt.show()

    
    
    
            
            
            