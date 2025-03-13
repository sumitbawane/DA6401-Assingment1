from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import wandb 
import keras 
from main import *

# Load and preprocess Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Split training data to create a validation set (10% of training data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)



optimizer_params = {
    'learning_rate': 0.001,
    'beta1':0.9,
    'beta2':0.999,
    'epsilon':1e-8
}

model=FeedForwardNeuralNetwork(
    layer_sizes=[784, 128, 128,128,128,128, 10],
    epochs=10,
    init_method='xavier',
    batch_size=64,
    optimizer='nadam',
    hidden_activation='tanh',
    **optimizer_params)

y_train_one_hot = model.oneHotEncoder(y_train)
y_val_one_hot = model.oneHotEncoder(y_val)
y_test_one_hot = model.oneHotEncoder(y_test)
model.train(X_train_flat, y_train_one_hot, epochs=10, batch_size=64, learning_rate=0.001, optimizer='nadam')

y_pred = model.predict(X_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)

# Add this function to your script
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix and return the figure.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
    
    Returns:
        matplotlib figure
    """
    # Convert one-hot encoded labels back to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return plt.gcf()
    



fashion_mnist_classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
img=plot_confusion_matrix(y_test, y_pred_classes,fashion_mnist_classes)
# Then add this code block at the end of your train_model_with_wandb function,
# right after calculating the test accuracy and before the final print:

# Log confusion matrix to wandb
wandb.login()
wandb.init(project="da6401-Assignment1")
wandb.log({"confusion_matrix": wandb.Image(img)})
