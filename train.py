import argparse
import numpy as np
import wandb
import keras
import sys
import os
from util import *
from loss import *
# Import your neural network implementation
from main import FeedForwardNeuralNetwork
from sklearn.model_selection import train_test_split

def preprocess_data(X_train, y_train, X_test, y_test,X_val,y_val, num_classes):
    """Preprocess the data for training and testing"""
    # Normalize the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_val=X_val.astype('float32')/255.0
    
    # Reshape the data (for MNIST/Fashion MNIST)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    # Convert labels to one-hot encoding
    
    y_train_one_hot=oneHotEncoder(y_train)
    y_val_one_hot=oneHotEncoder(y_val)
    y_test_one_hot=oneHotEncoder(y_test)
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot,X_val,y_val_one_hot

def get_optimizer_params(args):
    """Return the appropriate optimizer parameters based on command line arguments"""
    optimizer_params = {
        "learning_rate": args.learning_rate
    }
    
    # Add optimizer-specific parameters
    if args.optimizer in ["momentum", "nag"]:
        optimizer_params["momentum"] = args.momentum
    elif args.optimizer == "rmsprop":
        optimizer_params["decay_rate"] = args.beta
        optimizer_params["epsilon"]=args.epsilon
    elif args.optimizer in ["adam", "nadam"]:
        optimizer_params["beta1"] = args.beta1
        optimizer_params["beta2"] = args.beta2
        optimizer_params["epsilon"]=args.epsilon
    
    return optimizer_params

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train feedforward neural network with various configurations')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', default='da6401-Assignment1', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='', help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset and training arguments
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network')
    
    # Loss and optimizer arguments
    parser.add_argument('-l', '--loss', default='cross_entropy_loss', choices=['mean_loss', 'cross_entropy_loss'], help='Loss function for training')
    parser.add_argument('-o', '--optimizer', default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.99, help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001, help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay used by optimizers')
    
    # Model architecture arguments
    parser.add_argument('-w_i', '--weight_init', default='xavier', choices=['random', 'Xavier'], help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of neurons in hidden layers')
    parser.add_argument('-a', '--activation', default='tanh', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function for hidden layers')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.login()
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "beta": args.beta,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "weight_decay": args.weight_decay,
            "weight_init": args.weight_init,
            "num_hidden_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation
        }
    )
    
    # Load dataset
    if args.dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        num_classes = 10
    elif args.dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Preprocess data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train, y_train, X_test, y_test,X_val,y_val = preprocess_data(X_train, y_train, X_test, y_test,X_val,y_val ,num_classes)
    # Define layer dimensions
    input_dim = X_train.shape[1]
    hidden_layers = [args.hidden_size] * args.num_layers
    layer_sizes = [input_dim] + hidden_layers + [num_classes]
    
    # Get optimizer parameters
    optimizer_params = get_optimizer_params(args)
    
    # Create neural network using your FeedForwardNeuralNetwork class
    model = FeedForwardNeuralNetwork(
        layer_sizes=layer_sizes,
        optimizer=args.optimizer,
        hidden_activation=args.activation,
        init_method=args.weight_init,
        epochs=args.epochs,
        batch_size=args.batch_size,
        **optimizer_params
    )
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    def train_and_log(model, X_train, y_train, X_test, y_test):
        
        # Create descriptive run name
        run_name = f"hl_{args.num_layers}_bs_{args.batch_size}_lr_{args.learning_rate}_opt_{args.optimizer}_act_{args.activation}_wd_{args.weight_decay}_init_{args.weight_init}"
        wandb.run.name = run_name
        # Training loop - reimplement here to add wandb logging
        for epoch in range(args.epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            
            for i in range(0, X_train.shape[0], args.batch_size):
                # Get batch
                X_batch = X_shuffled[i:i + args.batch_size]
                y_batch = y_shuffled[i:i + args.batch_size]
                
                # Forward pass
                output = model.forward_pass(X_batch)
                
                # Compute gradients
                gradients_w, gradients_b = model.back_pass(X_batch, y_batch)
                
                # Update weights
                model.update_weights(gradients_w, gradients_b)
                
            train_pred = model.forward_pass(X_train)
            train_loss = cross_entropy_loss(train_pred, y_train)
            train_accuracy = accuracy(train_pred, y_train)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Evaluate on validation data
            val_pred = model.forward_pass(X_val)
            val_loss = cross_entropy_loss(val_pred, y_val)
            val_accuracy = accuracy(val_pred, y_val)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Log metrics to wandb
            wandb.log({
                "dataset":args.dataset,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                    
    
    
    train_and_log(model, X_train, y_train, X_test, y_test)
    
    test_pred = model.forward_pass(X_test)
    test_loss = cross_entropy_loss(test_pred, y_test)
    test_accuracy = accuracy(test_pred, y_test)
    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "loss": test_loss,

    })  
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    

    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()