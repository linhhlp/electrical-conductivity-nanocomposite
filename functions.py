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
import tensorflow as tf
import matplotlib.pyplot as plt
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
    
    
### References: https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5
###             https://www.tensorflow.org/guide/keras/custom_callback
from IPython.display import clear_output
class plotRealTime(tf.keras.callbacks.Callback):
    """ Plot Loss and Accuracy while training (graph real time fitting)"""
    def __init__(self, step=100):
        self.step = step
        self.epoch = 0
        self.N = 0
    
    def saveData( self, logs ):
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
                
    def plotData(self, logs):
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)
        
        if self.epoch % self.step == 0: 
            x_range = [self.step * x for x in range(0, self.N)]
        else:
            x_range = [self.step * x for x in range(0, self.N-1)]
            x_range.append(self.epoch)

        for i, metric in enumerate(metrics):
            axs[i].plot(x_range, 
                        self.metrics[metric], 
                        label=metric, 
                        marker="+")
            if logs['val_' + metric]:
                axs[i].plot(x_range, 
                            self.metrics['val_' + metric], 
                            label='val_' + metric,
                            marker="o")
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()
        
    def on_train_begin(self, logs=None):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            
    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        step = self.step
        self.epoch += 1
        if epoch % step != 0: return None;
        self.N = epoch // step + 1
        self.saveData(logs)
        self.plotData(logs)
        
    def on_train_end(self, logs=None):
        # Force to replot including last point of data
        self.saveData(logs)
        self.N += 1        
        self.plotData(logs)        


# patience: Number of epochs with no improvement after which training will be stopped.
# monitor: Quantity to be monitored.
def earlyStopper(patience=100):
    return tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                verbose=1)

def reduceLR(patience=100, min_lr=0.000001):
    return tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=patience, min_lr=min_lr)


def modelCheckpointCallback(checkpoint_filepath = 'tmp/checkpoint'):
    return tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
                
def smoothArray(data, size = 10):
    """ Smooth data in 1-D array 
    Data acts like signal with noise, fluctuates 
    References: https://www.delftstack.com/howto/python/smooth-data-in-python/
    """
    new_arr = np.convolve(data, np.ones(size) / size, mode='same')
    new_arr[0:5] = np.average(data[0:5] )
    new_arr[-5:]  = np.average(data[-5] )
    return new_arr
