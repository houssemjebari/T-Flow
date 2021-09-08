'''
Author: Jebari Houssem
University: National Institute of Applied Science and Technology (INSAT)
Date: September 8, 2021

This is an Open Source project that builds a neural networks library 
from scratch that mimicks the Keras framework. Anyone is free to 
contribute to this project to make it stand out. 
For more information read the Readme file in my Github repository.

'''

import numpy as np

class Loss:   
    '''
    This class is the base class for all the loss functions.
    '''
    def calculate_loss(self,y_pred,y_true):
       # calculates the loss function
       pass
    def calculate_grad(self,y_pred,y_true):
        # calculates the loss gradient
        pass

class CategoricalCrossEntrpoy(Loss):
    
    '''
    Categorical cross entropy loss function
    '''
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        confidence = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood = -np.log(confidence)
        return negative_log_likelihood


class BinaryCrossEntropy(Loss):

    '''
    Binary cross entropy loss function
    '''
    def calculate_loss(self,y_true,y_pred):
        return -np.mean(y_true * np.log(y_pred)  + (1 - y_true) * np.log(1 - y_pred))
    
    def calculate_grad(self,y_true,y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class MSE(Loss):

    '''
    Mean squared error loss function
    ''' 
    def calculate_loss(self,y_true,y_pred):
        return np.mean(np.power(y_true-y_pred,2))

    def calculate_grad(self,y_true,y_pred):
        return 2 * ((y_pred - y_true) / np.size(y_true))


    