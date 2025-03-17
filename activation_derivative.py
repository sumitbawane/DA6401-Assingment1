import numpy as np

def activation_derivative( type, x):
    if type == 'sigmoid':
        return x * (1 - x)
    elif type == 'relu':
        return 1.0 * (x > 0)
    elif type == 'tanh':
       
        return 1 - x**2
    elif type == 'softmax':
        return x * (1 - x)  
    else:
        return x