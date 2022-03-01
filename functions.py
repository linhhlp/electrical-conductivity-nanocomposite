#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# ---------------------------------------------------------------------------
""" Some basic functions for model of electrical conductivity""" 
# ---------------------------------------------------------------------------

__author__      = "Linh Hoang (linhhlp)"
__copyright__   = "Copyright 2022, Project: Prediction of Electrical Conductivity"
__license__ = "MIT"
__version__ = "1.0.1"

import numpy as np
from   sklearn.preprocessing import MinMaxScaler, StandardScaler

def safe_log10(data):
    """ A safe function for LOG10 in cases of too small values (close or equal to 0 ) """
    prob_tmp  = np.where(data > 1.0e-10, data, 1.0e-10)
    result    = np.where(data > 1.0e-10, np.log10(prob_tmp), -10)
    return result
    
class superHighVariationScaler:
    """ Standard scaler of Y with multiple scaler 
    Because the range is to high => log10 (values) to play with magnitude order
    To make it organize, use another Scaler following LOG10.
    Default scale is StandardScaler()
    """
    def __init__(self, scaler = None):
        if scaler:
            self.Scaler  = scaler
        else:
            self.Scaler  = StandardScaler()
    def fit_transform(self, data):
        data_loged = safe_log10(data)
        return self.Scaler.fit_transform (data_loged)
    def transform(self, data):
        data_loged = safe_log10(data)
        return self.Scaler.transform (data_loged)
    
    def inverse_transform(self,data):
        data_unloged = self.Scaler.inverse_transform (data)
        return np.float_power(10, data_unloged )


class superHighVariationScalerSimple:
    """ Simple Scaler with only LOG10 """
    def fit_transform(self, data):
        return safe_log10(data)

    def transform(self, data):
        return safe_log10(data)
    
    def inverse_transform(self,data):
        return np.float_power(10, data )

class noScaler:
    """ Just return the same data """
    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data
    
    def inverse_transform(self,data):
        return data

    
        
def mapStringToNum (data):
    polymer_map_list =  {'HDPE':0,'HDPEtreated':1}
    filler_map_list =   {'MWCNT':0,'SWCNT':1,'GNP':2}
    data['polymer_1'] = data['polymer_1'].map(polymer_map_list)
    data['filler_1']  = data['filler_1' ].map(filler_map_list)
    return data

def mapNumToString (data):
    polymer_map_list =  {0:'HDPE',1:'HDPEtreated'}   # inversed_map = {v: k for k, v in polymer_map_list.items()}
    filler_map_list =   {0:'MWCNT',1:'SWCNT',2:'GNP'}
    data['polymer_1'] = data['polymer_1'].map(polymer_map_list)
    data['filler_1']  = data['filler_1' ].map(filler_map_list)
    return data