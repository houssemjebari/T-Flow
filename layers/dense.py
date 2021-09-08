import numpy as np
from layer import Layer

class Dense(Layer):
    
    '''
    Most classical neural networks layers. It can be used with structured data or as 
    a classifier or a regressor on top of a feature extractor.
    '''
    def __init__(self,input_size,output_size):
        super().__init__()
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)

    def forward(self,input):
        self.input = input
        self.output = (np.dot(self.weights,input) + self.bias)
        return self.output     
    
    def backward(self,output_gradient,learning_rate):
        weights_gradient = np.dot(output_gradient,self.input.T) 
        X_gradient = np.dot(self.weights.T,output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return X_gradient
