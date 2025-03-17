import numpy as np
def activations(type, x):
        if type == 'sigmoid':
           
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif type == 'relu':
            return np.maximum(0, x)
        elif type == 'tanh':
            
            x = np.clip(x, -500, 500)
            return np.tanh(x)
        elif type == 'softmax':
            
            x = np.clip(x, -500, 500)
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x