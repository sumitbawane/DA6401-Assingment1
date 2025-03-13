import numpy as np
def cross_entropy_loss( y_hat, y):
    m = y.shape[0]
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss


def squared_mean_erro(y_hat,y):
    pass