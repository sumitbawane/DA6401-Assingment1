import numpy as np
from activations import * 
from activation_derivative import *
from loss import *
from util import * 
from optimizers import *

class FeedForwardNeuralNetwork:
    def __init__(self, layer_sizes,optimizer='sgd', loss_function='cross_entropy_loss',hidden_activation='sigmoid', init_method='random',epochs=32,batch_size=32, **optimizer_params):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.hidden_activation = hidden_activation
        self.epochs=epochs
        self.batch_size=batch_size
        self.optimizer=optimizer
        self.loss_function=loss_function
        # Print layer sizes for debugging
        print(f"Initializing network with layer sizes: {layer_sizes}")
        print(f"Using activation function: {hidden_activation}")
        print(f"Using initialization method: {init_method}")

        for i in range(len(layer_sizes) - 1):
            if init_method.lower() == 'xavier':
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            else:  
                weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01

            biases = np.zeros(layer_sizes[i + 1])

            print(f"Layer {i}: Weight shape = {weights.shape}, Bias shape = {biases.shape}")

            self.weights.append(weights)
            self.biases.append(biases)

        # Set up the self.optimizer
        self.setup_optimizer(self.optimizer, **optimizer_params)

    def setup_optimizer(self, optimizer='sgd', **optimizer_params):
        optimizer_map = {
            'sgd': SGD,
            'momentum': Momentum,
            'nag': NesterovAcceleratedGradient,
            'rmsprop': RMSProp,
            'adam': Adam,
            'nadam': NAdam
        }

        if self.optimizer.lower() in optimizer_map:
            self.optimizer = optimizer_map[self.optimizer.lower()](**optimizer_params) #object of the optimizer class
        else:
            print(f"Warning: Unknown self.optimizer '{self.optimizer}'. Defaulting to SGD.")
            self.optimizer = SGD(**optimizer_params) #default 

        # Initialize self.optimizer state
        self.optimizer.initialize(self.weights, self.biases)

    def forward_pass(self, x):
        self.a_values = []
        self.h_values = [x]

        for i in range(len(self.weights)):
            if self.h_values[i].shape[1] != self.weights[i].shape[0]:
                raise ValueError(f"Matrix dimension mismatch: h_values[{i}] has shape {self.h_values[i].shape}, weights[{i}] has shape {self.weights[i].shape}")

            self.a_values.append(np.dot(self.h_values[i], self.weights[i]) + self.biases[i])

            if i == len(self.weights) - 1:
                self.h_values.append(activations("softmax", self.a_values[i]))
            else:
                self.h_values.append(activations(self.hidden_activation, self.a_values[i]))

        return self.h_values[-1]  

    def back_pass(self, x, y):
        """
        Computes gradients for backpropagation but doesn't update weights directly.
        Returns gradients for the self.optimizer to use.
        """
        m = x.shape[0]
        y_hat = self.h_values[-1]

        if(self.loss_function=='cross_entropy_loss'):
            delta=y_hat-y
        elif(self.loss_function=='mse_loss'):
            delta=mse_loss_derivative(y_hat,y)

        # Initialize lists to store gradients
        gradients_w = []
        gradients_b = []

        # Calculate gradients for the output layer
        dw = np.dot(self.h_values[-2].T, delta) / m
        db = np.sum(delta, axis=0) / m

        # Store gradients 
        gradients_w.append(dw)
        gradients_b.append(db)

        # Backpropagation for hidden layers
        current_delta = delta
        for i in range(len(self.weights) - 2, -1, -1):
            current_delta = np.dot(current_delta, self.weights[i + 1].T) * activation_derivative(self.hidden_activation, self.h_values[i + 1])
            dw = np.dot(self.h_values[i].T, current_delta) / m
            db = np.sum(current_delta, axis=0) / m

            # Store gradients (in reverse order)
            gradients_w.append(dw)
            gradients_b.append(db)

        # Reverse lists to match layer ordering
        gradients_w.reverse()
        gradients_b.reverse()

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        """Update weights using the configured self.optimizer"""
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, gradients_w, gradients_b
        )
        
    def train(self, x, y, **optimizer_params):
        loss=[]
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            print(f"Reshaped input data to {x.shape}")

        if y.shape[0] != x.shape[0]:
            raise ValueError(f"Number of samples in x ({x.shape[0]}) and y ({y.shape[0]}) must match")

        # Print shapes for debugging
        print(f"Training data shapes: x: {x.shape}, y: {y.shape}")
        print(f"First layer input size: {self.layer_sizes[0]}")

        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Input feature dimension {x.shape[1]} doesn't match first layer size {self.layer_sizes[0]}")

        # Print self.optimizer info
        print(f"Training with self.optimizer: {type(self.optimizer).__name__}")

        for epoch in range(self.epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, x.shape[0], self.batch_size):
                x_batch = x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Debug: print batch shapes
                if epoch == 0 and i == 0:
                    print(f"First batch shapes: x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")

                # Forward pass
                output = self.forward_pass(x_batch)

                # Debug: print output shape
                if epoch == 0 and i == 0:
                    print(f"First batch output shape: {output.shape}")

                # Compute gradients
                gradients_w, gradients_b = self.back_pass(x_batch, y_batch)

                # Update weights using the self.optimizer self.update weights calls self.optimizer.update_weights
                self.update_weights(gradients_w, gradients_b)

            # Evaluate on a subset of the data to save time
            eval_indices = np.random.choice(x.shape[0], min(1000, x.shape[0]), replace=False)
            x_eval = x[eval_indices]
            y_eval = y[eval_indices]

            prediction = self.forward_pass(x_eval)
            if(self.loss_function=='cross_entropy_loss'):
                loss.append(cross_entropy_loss(prediction, y_eval))
                loss_=cross_entropy_loss(prediction, y_eval)
            else:
                loss.append(mse_loss(prediction,y_eval))
                loss_=cross_entropy_loss(prediction, y_eval)
            accuracy_ = accuracy(prediction, y_eval)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss_:.4f}, Accuracy: {accuracy_:.4f}")
        return loss,self.weights,self.biases
    def predict(self, x):
        return self.forward_pass(x)

  
