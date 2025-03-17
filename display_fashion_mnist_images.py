import wandb
import numpy as np
import pandas as pd
import keras 

# Initialize WandB
wandb.init(project="da6401-Assignment1")

# Load dataset
(x_train, y_train), (x_test,y_test) = keras.datasets.fashion_mnist.load_data()


df = pd.DataFrame({"image": list(x_train), "label": y_train})

for label, group in df.groupby("label"):
    image = group.iloc[0]["image"]  # Get the first image of each class
    wandb.log({"fashion_mnist_samples": wandb.Image(image, caption=f"Label {label}")})

# Finish WandB run
wandb.finish()
