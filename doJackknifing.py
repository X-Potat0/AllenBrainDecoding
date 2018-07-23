"""
Created on Fri Jul 20 15:22:06 2018
Get decoding contribution per neuron using jackknifing
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Settings
num_splits = 5
group_size = 15
num_it = 500
temp_freq = 1
# cre_line = 'Emx1-IRES-Cre'
cre_line = 'Cux2-CreERT2'
area = 'VISp'

# Set up decoder
kf = KFold(n_splits=num_splits)
gnb = GaussianNB()

def bayesian_decoding(resp, stim, neuron_id):
    perf = 0
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        gnb.fit(train_resp[:, neuron_id], stim[train_index])
        y_pred = gnb.predict(test_resp[:, neuron_id])
        perf = perf + np.sum(stim[test_index] == y_pred) / len(y_pred)
    perf = perf / num_splits
    return perf


# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=[area], cre_lines=[cre_line])

# Loop through recordings


for i in range(0, len(ecs)):
# for i in range(0, 1):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    resp_mat = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')

    # Select trials
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)]

    # Loop over neurons
    contr_mat = np.empty([num_it, len(resp_mat[0])])
    for n in range(len(resp_mat[0])):
        for j in range(num_it):
            # Get subsample and decode with and without neuron n
            neurons = np.append(n, np.random.choice(len(decode_resp[0]), group_size-1, replace=False))
            perf_all = bayesian_decoding(decode_resp, decode_ori, neurons)
            perf_excl = bayesian_decoding(decode_resp, decode_ori, neurons[1:len(neurons)])
            contr_mat[j,n] = group_size*perf_all-(group_size-1)*perf_excl
    contr = np.mean(contr_mat, axis=0)
    np.save('/home/guido/Projects/Orient/boc/ophys_processed/' + str(exp['id']) + '_neuron_contribution', contr)







