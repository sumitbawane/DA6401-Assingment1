import numpy as np


def oneHotEncoder(y):
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




def accuracy(predictions, labels):
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    return accuracy
