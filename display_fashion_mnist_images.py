import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import numpy as np
import wandb

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Initialize WandB
wandb.init(project="da6401-Assignemt1")

# Create a grid to plot sample images for each class
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
axes = axes.flatten()

# Iterate through each class
for i in range(10):
    # Find index of the first image for the current class
    class_idx = np.where(y_train == i)[0][0]
    img = x_train[class_idx]

    # Plot the image
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')

# Log the plot to WandB
wandb.log({"sample_images": wandb.Image(fig)})

plt.show()
