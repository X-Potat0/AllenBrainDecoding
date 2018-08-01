"""
Created on Fri Jul 20 15:22:06 2018
Perform decoding of orientation using different sizes of random subsets
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
num_splits = 5
group_sizes = range(5,101,5)
num_it = 500
temp_freq = 1
cre_line = ['Emx1-IRES-Cre', 'Cux2-CreERT2']
# cre_line = ['Cux2-CreERT2']
area = ['VISp']
root_path = get_path()

# Get datasets
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
ecs = boc.get_experiment_containers(targeted_structures=area, cre_lines=cre_line)

decode_perf = pd.DataFrame()
for i in range(0, len(ecs)):
# for i in range(0, 2):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    resp_mat = np.load(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.npy')
    stim_data = pd.read_pickle(root_path + 'boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    
    # Select trials
    decode_ori = np.array(stim_data.orientation[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)])
    decode_resp = resp_mat[(pd.notnull(stim_data.orientation)) & (stim_data.temporal_frequency == temp_freq)]
    
    all_perf = pd.DataFrame()
    for n in group_sizes:
        print('Group size ' + str(n))
        perf = np.array([])
        for j in range(num_it):
            neurons = np.random.choice(len(decode_resp[0]), n, replace=False)
            this_perf = bayesian_decoding(decode_resp, decode_ori, neurons, num_splits)
            perf = np.append(perf, this_perf)
        all_perf[str(n)] = perf
        
    decode_perf[exp['id']] = np.mean(all_perf, axis=0)
   
decode_perf.to_pickle(root_path + 'boc/ophys_decoding_results/BayesianDecoding_' + cre_line[0] + '_' + area[0] + '.pkl')

# Plot output
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for i in list(decode_perf):
    ax1.plot(decode_perf[i])
ax1.set_ylabel('Decoding performance')
ax1.set_xlabel('Neurons')
ax1.set_xticks(np.arange(1, 20, step=2))

ax2.plot(np.mean(decode_perf, axis=1))
std_err = np.std(decode_perf, axis=1)/np.sqrt(len(list(decode_perf)))
ax2.fill_between(list(decode_perf.index.values), np.mean(decode_perf, axis=1)-std_err, np.mean(decode_perf, axis=1)+std_err, alpha=0.4)
ax2.set_xlabel('Neurons')
ax2.set_xticks(np.arange(1, 20, step=2))

plt.tight_layout()
plt.savefig('/home/guido/Projects/AllenBrainDecoding/Plots/BayesianDecoding/SampleSizeDecoding_' + area[0])
# plt.savefig(root_path + 'Plots/BayesianDecoding/SampleSizeDecoding_' + cre_line[0] + '_' + area[0])
plt.show()
    



        
        
    
    




