import numpy as np
def cross_entropy_loss( y_hat, y):
    m = y.shape[0]
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss


def mse_loss(y_hat, y):
    m = y.shape[0]
    loss = np.sum((y_hat - y) ** 2) / (2 * m)
    return loss
    
def mse_loss_derivative(y_hat,y):
    m=y.shape[0]
    return 2*(y_hat-y)/m