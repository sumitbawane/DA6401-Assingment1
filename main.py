import numpy as np 
import pandas as pd 
import keras


(x_train,y_train) , (x_test,y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



class FeedForwardNeuralNetwork:
    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights=[]
        self.biases=[]
        for i in range(len(layer_sizes)-1):
            weights=np.random.randn(layer_sizes[i],layer_sizes[i+1])
            biases=np.random.randn(layer_sizes[i+1])
            self.weights.append(weights)
            self.biases.append(biases)


layer_sizes = [x_train.shape[1], 128, 64, 10]
model= FeedForwardNeuralNetwork(layer_sizes)

print(model.biases)
print(model.weights)