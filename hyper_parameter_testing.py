import numpy as np
import wandb
import keras
from main import *
from util import *
from loss import *
from sklearn.model_selection import train_test_split




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

def train_model_with_wandb():
    run = wandb.init()
    config = wandb.config

    # descriptive run name
    run_name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_lr_{config.learning_rate}_opt_{config.optimizer}_act_{config.activation}_wd_{config.weight_decay}_init_{config.weight_init}_lf_{config.loss_function}"
    wandb.run.name = run_name


    # Build layer sizes based on config
    input_size = 28 * 28  # Flattened Fashion MNIST
    hidden_layers = [config.layer_size] * config.hidden_layers
    layer_sizes = [input_size] + hidden_layers + [10]  # 10 output classes

    # Prepare optimizer parameters based on the chosen optimizer
    optimizer_params = {
        'learning_rate': config.learning_rate
    }

    # Add specific parameters for each optimizer type
    if config.optimizer in ['momentum', 'nag']:
        optimizer_params['momentum'] = 0.9
    elif config.optimizer in ['rmsprop']:
        optimizer_params['decay_rate'] = 0.9
        optimizer_params['epsilon'] = 1e-8
    elif config.optimizer in ['adam', 'nadam']:
        optimizer_params['beta1'] = 0.9
        optimizer_params['beta2'] = 0.999
        optimizer_params['epsilon'] = 1e-8

    # Create model with configured optimizer
    model = FeedForwardNeuralNetwork(
        layer_sizes=layer_sizes,
        optimizer=config.optimizer,
        loss_function=config.loss_function,
        hidden_activation=config.activation,
        init_method=config.weight_init,
        epochs=config.epochs,
        batch_size=config.batch_size,
        **optimizer_params
    )

    # Convert labels to one-hot encoding
    y_train_one_hot = oneHotEncoder(y_train)
    y_val_one_hot = oneHotEncoder(y_val)
    y_test_one_hot = oneHotEncoder(y_test)

    # Store metrics per epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop with validation after each epoch
    for epoch in range(config.epochs):
       
        indices = np.random.permutation(X_train_flat.shape[0])
        X_shuffled = X_train_flat[indices]
        y_shuffled = y_train_one_hot[indices]

        
        for i in range(0, X_train_flat.shape[0], config.batch_size):
            x_batch = X_shuffled[i:i + config.batch_size]
            y_batch = y_shuffled[i:i + config.batch_size]

            # Forward pass
            output = model.forward_pass(x_batch)

            # Compute gradients and update weights
            gradients_w, gradients_b = model.back_pass(x_batch, y_batch)

           
            if config.weight_decay > 0:
                for j in range(len(gradients_w)):
                    gradients_w[j] += config.weight_decay * model.weights[j]

            model.update_weights(gradients_w, gradients_b)

        # Evaluate on training data
        train_pred = model.forward_pass(X_train_flat)
        if(model.loss_function=='cross_entropy_loss'):
            train_loss = cross_entropy_loss(train_pred, y_train_one_hot)
        elif(model.loss_function=='mse_loss'):
            train_loss=mse_loss(train_pred,y_train_one_hot)
            
        
        train_accuracy = accuracy(train_pred, y_train_one_hot)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate on validation data
        val_pred = model.forward_pass(X_val_flat)
        if(model.loss_function=='cross_entropy_loss'):
            val_loss = cross_entropy_loss(val_pred, y_val_one_hot)
        elif(model.loss_function=='mse_loss'):
            val_loss=mse_loss(val_pred,y_val_one_hot)
        val_accuracy = accuracy(val_pred, y_val_one_hot)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Log metrics to wandb
        wandb.log({
            "loss_function":model.loss_function,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    
    test_pred = model.forward_pass(X_test_flat)
    test_loss = cross_entropy_loss(test_pred, y_test_one_hot)
    test_accuracy = accuracy(test_pred, y_test_one_hot)
    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "loss": test_loss,

    })

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        's': 2,
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'loss_function': {
            'values':['mse_loss','cross_entropy_loss']
        },
        'hidden_layers': {
            'values': [3, 4, 5]
        },
        'layer_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_init': {
            'values': ['random', 'xavier']
        },
        'activation': {
            'values': ['sigmoid', 'relu', 'tanh']
        }
    }
}

# Initialize wandb
wandb.login()

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="da6401-Assignment1")

# Run the sweep
wandb.agent(sweep_id, train_model_with_wandb, count=20)  

