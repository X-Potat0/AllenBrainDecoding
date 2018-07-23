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

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
for i in range(0, len(ecs)):
# for i in range(0, 1):
    # Load in data
    print('Decoding recording ' + str(i) + ' of ' + str(len(ecs)))
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    cells = pd.read_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_cells.pkl')
    neuron_contr = np.load('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_neuron_contribution.npy')
    
    # Add neuron contribution to cell dataframe
    cells = cells.assign(bayes_contribution=neuron_contr)
    
    # Plot
    ax1.scatter(cells['osi_dg'], cells['bayes_contribution'])
    ax2.scatter(cells['dsi_dg'], cells['bayes_contribution'])
    ax3.scatter(cells['reliability_dg'], cells['bayes_contribution'])
    ax4.scatter(cells['tfdi_dg'], cells['bayes_contribution'])

ax1.set_ylabel('Decoding contribution')
ax1.set_xlabel('Orientation selectivity')
ax2.set_xlabel('Direction selectivity')
ax3.set_xlabel('Peak dF/F')
ax4.set_xlabel('TF discrimination index')
ax3.set_ylabel('Decoding contribution')
plt.tight_layout()
plt.show()
    
    
    
    
    
    
    

