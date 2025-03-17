
## DA6401 ASSIGNMENT1 

## Files and Structure

- `main.py`: Core neural network implementation with the `FeedForwardNeuralNetwork` class
- `activations.py`: Implementation of activation functions
- `activation_derivative.py`: Derivatives of activation functions for backpropagation
- `loss.py`: Loss functions and their derivatives
- `optimizers.py`: Implementation of various optimization algorithms
- `util.py`: Utility functions including one-hot encoding and accuracy calculation
- `train.py`: Command-line interface for training models with various configurations
- `hyper_parameter_testing.py`: Hyperparameter optimization using Weights & Biases sweeps
- `plot_confusion_matrix.py`: Confusion matrix visualization for model evaluation
- `display_fashion_mnist_images.py`: Utility to visualize Fashion MNIST dataset samples

## Getting Started

### Prerequisites

- Python 3.10 (versions greater than 3.10 were not compatible with tensorflow)
- NumPy
- Keras & tensorflow (for dataset loading) (I had to install them seprately)
- Scikit-learn (for just splitting the train data into validation and training )
- Matplotlib (for plotting confusion matrix)
- Seaborn (for plotting confusion matrix)
- Weights & Biases (to log various parameter's to generate plots)

### Installation

```bash
pip install numpy keras scikit-learn matplotlib seaborn wandb tensorflow
```

### Training a Model

You can train a model using the command-line interface:

```bash
python train.py --dataset fashion_mnist --epochs 10 --batch_size 64 --optimizer nadam --activation tanh --num_layers 5 --hidden_size 128
```
Note : By default all the parameter's are set to the best performing model

### Command Line Arguments

The following command-line arguments are supported, along with their default values:

#### Weights & Biases Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | `da6401-Assignment1` | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | `''` | Wandb Entity used to track experiments in the Weights & Biases dashboard |

#### Dataset and Training Parameters
| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `-d`, `--dataset` | `fashion_mnist` | `mnist`, `fashion_mnist` | Dataset to use for training |
| `-e`, `--epochs` | `10` | - | Number of epochs to train neural network |
| `-b`, `--batch_size` | `64` | - | Batch size used to train neural network |

#### Loss Function and Optimizer
| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `-l`, `--loss` | `cross_entropy_loss` | `mse_loss`, `cross_entropy_loss` | Loss function for training |
| `-o`, `--optimizer` | `nadam` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | Optimizer to use |

#### Optimizer Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `-lr`, `--learning_rate` | `0.001` | Learning rate used to optimize model parameters |
| `-m`, `--momentum` | `0.5` | Momentum used by momentum and nag optimizers |
| `-beta`, `--beta` | `0.9` | Beta used by rmsprop optimizer |
| `-beta1`, `--beta1` | `0.9` | Beta1 used by adam and nadam optimizers |
| `-beta2`, `--beta2` | `0.99` | Beta2 used by adam and nadam optimizers |
| `-eps`, `--epsilon` | `0.000001` | Epsilon used by optimizers |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay used by optimizers (L2 regularization) |

#### Model Architecture
| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `-w_i`, `--weight_init` | `xavier` | `random`, `Xavier` | Weight initialization method |
| `-nhl`, `--num_layers` | `5` | - | Number of hidden layers in feedforward neural network |
| `-sz`, `--hidden_size` | `128` | - | Number of neurons in hidden layers |
| `-a`, `--activation` | `tanh` |  `sigmoid`, `tanh`, `ReLU` | Activation function for hidden layers |

## Hyperparameter Optimization

To perform hyperparameter optimization using Weights & Biases sweeps:

```bash
python hyper_parameter_testing.py
```

This script uses Bayesian optimization to find optimal hyperparameters for the neural network.

## Visualization

To visualize the confusion matrix for a trained model:

```bash
python plot_confusion_matrix.py
```

To display sample images from the Fashion MNIST dataset:

```bash
python display_fashion_mnist_images.py
```

## Best Performing model

The model can achieve good performance on the Fashion MNIST dataset with appropriate hyperparameters:
- Using NAdam optimization
- Xavier initialization
- Tanh activation
- 5 hidden layers with 128 neurons each
- Batch size of 64
- Learning rate of 0.001

## Example Usage

```python
from main import *
import numpy as np
import keras
from util import *

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_one_hot = oneHotEncoder(y_train)
y_test_one_hot = oneHotEncoder(y_test)

# Create and train model
model = FeedForwardNeuralNetwork(
    layer_sizes=[784, 128, 128, 128, 128, 128, 10],
    optimizer='nadam',
    hidden_activation='tanh',
    init_method='xavier',
    epochs=10,
    batch_size=64,
    learning_rate=0.001
)

model.train(X_train_flat, y_train_one_hot)

# Evaluate model
predictions = model.predict(X_test_flat)
from util import accuracy
acc = accuracy(predictions, y_test_one_hot)
print(f"Test accuracy: {acc:.4f}")
```
