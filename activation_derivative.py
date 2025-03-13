import numpy as np

def activation_derivative( type, x):
    if type == 'sigmoid':
        # The sigmoid derivative should be sigmoid(x)*(1-sigmoid(x))
        # But x is already the sigmoid output here
        return x * (1 - x)
    elif type == 'relu':
        return 1.0 * (x > 0)
    elif type == 'tanh':
        # tanh derivative is (1 - tanh²(x))
        return 1 - x**2
    elif type == 'softmax':
        return x * (1 - x)  # This is a simplification, only valid for certain cases
    else:
        return x