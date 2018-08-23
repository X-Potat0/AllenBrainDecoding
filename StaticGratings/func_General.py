"""
Created on Tue Jul 31 11:04:44 2018

General functions

@author: Guido Meijer
"""

import numpy as np

def stderr(input_data, axis=0):
    if len(np.shape(input_data)) == 1:
        std_err = np.std(input_data)/np.sqrt(len(input_data))
    elif len(np.shape(input_data)) == 2:
        std_err = np.std(input_data, axis=axis)/np.sqrt(np.shape(input_data)[axis])
    return std_err
        
        
    
