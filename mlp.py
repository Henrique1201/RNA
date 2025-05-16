import random
import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = np.random.randn(input_size, hidden_size)
        self.hidden_size = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)