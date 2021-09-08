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

class Layer:

    '''
    This class is the base class for all the neural network layers.
    '''
    def __init__(self):
        self.output = None
        self.input = None
    
    def forward(self,input):
        # Returns the output of the Layer
        pass

    def backward(self,output_gradient,learning_rate):
        # Updates parameters and returns input gradient
        pass





