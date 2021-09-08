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
from layer import Layer


class Activation(Layer):
    
    '''
    This is the base class for activation functions.
    '''
    def __init__(self,activation,activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self,input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self,output_gradient,learning_rate):
        return  np.multiply(output_gradient,self.activation_prime(self.input))


class Tanh(Activation):
    
    '''
    Hyperbolic tangent activation function 
    '''
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh,tanh_prime)


class Sigmoid(Activation):

    '''
    Sigmoid activation function
    '''
    def __init__(self):
        def sigmoid(x):
            return 1 / (1+np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid,sigmoid_prime)


class ReLu(Activation):

    '''
    Rectified Linear Unit activation function
    '''
    def __init__(self):
        relu = lambda x: np.maximum(0,x)
        relu_prime = lambda x: 1 if x > 0 else 0
        super().__init__(relu,relu_prime)


class Softmax(Activation):

    '''
    Softmax activation function
    '''
    def __init__(self):
        def softmax(x):
            exp_values = np.exp(x - np.max(x,axis=1,keepdims=True))
            probas = exp_values / np.sum(exp_values,axis=1,keepdims=True)
            return probas
        def softmax_prime(x):
            pass
        super().__init__(softmax,softmax_prime)

            
