import numpy as np 
from layer import Layer

class Flatten(Layer):
    
    '''
    Flattens a multidimensional layer into a 1D vector. It is often used to 
    flatten the output of the feature extractors and convnets and forms a bridge
    to use the dense layers.
    '''
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
    
    def forward(self,input):
        return np.reshape(input,(-1,1))
    
    def backward(self,output_gradient,learning_rate):
        return np.reshape(output_gradient,self.input_shape)
        