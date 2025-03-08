import numpy as np
import pandas as pd
import keras

class FeedForwardNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Print layer sizes for debugging
        print(f"Initializing network with layer sizes: {layer_sizes}")

        for i in range(len(layer_sizes) - 1):
            # Initialize weights with proper dimensions and scale them appropriately
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            biases = np.zeros(layer_sizes[i + 1])

            print(f"Layer {i}: Weight shape = {weights.shape}, Bias shape = {biases.shape}")
            self.weights.append(weights)
            self.biases.append(biases)

    def forward_pass(self, x):
      
        self.a_values = []
        self.h_values = [x]

        for i in range(len(self.weights)):
            # Debug dimension check
            if self.h_values[i].shape[1] != self.weights[i].shape[0]:
                raise ValueError(f"Matrix dimension mismatch: h_values[{i}] has shape {self.h_values[i].shape}, weights[{i}] has shape {self.weights[i].shape}")

            self.a_values.append(np.dot(self.h_values[i], self.weights[i]) + self.biases[i])

            if i == len(self.weights) - 1:
                self.h_values.append(self.activations("softmax", self.a_values[i]))
            else:
                self.h_values.append(self.activations("sigmoid", self.a_values[i]))

        return self.h_values[-1]  # Return the output

    def back_pass(self, x, y, learningRate):
        m = x.shape[0]
        y_hat = self.h_values[-1]

        delta = y_hat - y

        # Fixing the backpropagation for the output layer
        dw = np.dot(self.h_values[-2].T, delta) / m
        db = np.sum(delta, axis=0) / m
        self.weights[-1] -= learningRate * dw
        self.biases[-1] -= learningRate * db

        # Backpropagation for hidden layers
        current_delta = delta
        for i in range(len(self.weights) - 2, -1, -1):
            current_delta = np.dot(current_delta, self.weights[i + 1].T) * self.activation_derivative("sigmoid", self.h_values[i + 1])
            dw = np.dot(self.h_values[i].T, current_delta) / m
            db = np.sum(current_delta, axis=0) / m
            self.weights[i] -= learningRate * dw
            self.biases[i] -= learningRate * db
    
    def activations(self, type, x):
        if type == 'sigmoid':
            # Clip x to avoid overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif type == 'relu':
            return np.maximum(0, x)
        elif type == 'softmax':
            # Clip x to avoid overflow
            x = np.clip(x, -500, 500)
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            return x

    def activation_derivative(self, type, x):
        if type == 'sigmoid':
            # The sigmoid derivative should be sigmoid(x)*(1-sigmoid(x))
            # But x is already the sigmoid output here
            return x * (1 - x)
        elif type == 'relu':
            return 1 * (x > 0)
        elif type == 'softmax':
            return x * (1 - x)  # This is a simplification, only valid for certain cases
        else:
            return x

    def oneHotEncoder(self, y):
        # Function to convert labels to one-hot encoding
        # Ensure y is a 1D array
        y = np.array(y).reshape(-1)

        # Check the shape before processing
        print(f"Original y shape before one-hot encoding: {y.shape}")

        # Get number of classes
        num_classes = len(np.unique(y))
        if num_classes <= 10:  # Default to 10 classes if it's MNIST
            num_classes = 10

        # Create one-hot encoding
        one_hot = np.zeros((y.shape[0], num_classes))
        for i in range(y.shape[0]):
            one_hot[i][int(y[i])] = 1
        # Check the final shape
        print(f"One-hot encoded y shape: {one_hot.shape}")

        return one_hot

    def cross_entropy_loss(self, y_hat, y):
        m = y.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -np.sum(y * np.log(y_hat)) / m
        return loss

    def train(self, x, y, epochs, batch_size, learningRate):
        # Ensure x is properly flattened if it's image data
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            print(f"Reshaped input data to {x.shape}")

        # Verify that y has the right shape
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"Number of samples in x ({x.shape[0]}) and y ({y.shape[0]}) must match")

        # Print shapes for debugging
        print(f"Training data shapes: x: {x.shape}, y: {y.shape}")
        print(f"First layer input size: {self.layer_sizes[0]}")

        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Input feature dimension {x.shape[1]} doesn't match first layer size {self.layer_sizes[0]}")

        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for i in range(0, x.shape[0], batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Debug: print batch shapes
                if epoch == 0 and i == 0:
                    print(f"First batch shapes: x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")

                output = self.forward_pass(x_batch)

                # Debug: print output shape
                if epoch == 0 and i == 0:
                    print(f"First batch output shape: {output.shape}")

                self.back_pass(x_batch, y_batch, learningRate)

            # Evaluate on a subset of the data to save time
            eval_indices = np.random.choice(x.shape[0], min(1000, x.shape[0]), replace=False)
            x_eval = x[eval_indices]
            y_eval = y[eval_indices]

            prediction = self.forward_pass(x_eval)
            loss = self.cross_entropy_loss(prediction, y_eval)
            accuracy = self.accuracy(prediction, y_eval)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, x):
        return self.forward_pass(x)

    def accuracy(self, predictions, labels):
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy

# Example correct usage
# For MNIST data (28x28 pixel images, 10 classes):
# x_train should be of shape (num_samples, 784) after flattening
# y_train should be of shape (num_samples,) containing class labels 0-9
(x_train,y_train) , (x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
layer_sizes = [x_train.shape[1], 128, 64, 10]  # First layer MUST match input feature dimension
model = FeedForwardNeuralNetwork(layer_sizes)
y_train = y_train.reshape(-1)
y_train_one_hot = model.oneHotEncoder(y_train)
model.train(x_train, y_train_one_hot, epochs=10, batch_size=32, learningRate=0.01)