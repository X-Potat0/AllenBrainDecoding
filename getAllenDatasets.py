"""
Created on Fri Jul 20 15:22:06 2018
Get datasets from Allen Brain Observatory dataset and extract resonse per trial per neuron
@author: Guido Meijer
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd

# What feature to extract from fluorescence trace during stimulus
resp_operation = np.mean

# Get manifest json file
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

# Get experiment containers
targeted_structures = boc.get_all_targeted_structures()
cre_lines = boc.get_all_cre_lines()
ecs = boc.get_experiment_containers(targeted_structures=['VISp'], cre_lines=['Emx1-IRES-Cre'])

# Get all cell data
all_cells = boc.get_cell_specimens()
all_cells = pd.DataFrame.from_records(all_cells)

# Loop through recordings
for i in range(0, len(ecs)):
    # Get NWB file
    exp = boc.get_ophys_experiments(experiment_container_ids=[ecs[i]['id']], stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    data_set = boc.get_ophys_experiment_data(exp['id'])
    print('Processing recording ' + str(exp['id']) + ' [' + str(i+1) + ' of ' + str(len(ecs)) + ']')

    # Get fluo traces
    cell_ids = data_set.get_cell_specimen_ids()
    time, dff_traces = data_set.get_dff_traces(cell_specimen_ids=cell_ids)

    # Get stimulus times
    stim_times = data_set.get_stimulus_table(stimulus_name='drifting_gratings')

    # Extract mean or max per trial per neuron
    resp_mat = np.empty((len(stim_times), len(cell_ids)))
    for t in range(len(stim_times)):
        for n in range(len(cell_ids)):
            resp_mat[t,n] = resp_operation(dff_traces[n,range(stim_times.start[t],stim_times.end[t])])
            
    # Get cell properties for this exp
    cells = all_cells[all_cells['id'].isin(cell_ids)]
    
    # Save
    np.save('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']), resp_mat)
    stim_times.to_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_experiment_data/' + str(exp['id']) + '.pkl')
    cells.to_pickle('/home/guido/Projects/AllenBrainDecoding/boc/ophys_processed/' + str(exp['id']) + '_cells.pkl')

