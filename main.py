import numpy as np

class Optimizer:
    """Base optimizer class that implements update rule interface"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, weights, biases):
        """Initialize optimizer-specific parameters"""
        pass

    def update(self, weights, biases, gradients_w, gradients_b):
        """Update weights and biases based on gradients"""
        raise NotImplementedError("Subclasses must implement update method")

class SGD(Optimizer):
    """Standard Stochastic Gradient Descent"""
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients_w[i]
            biases[i] -= self.learning_rate * gradients_b[i]
        return weights, biases

class Momentum(Optimizer):
    """Momentum-based Gradient Descent"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

    def initialize(self, weights, biases):
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.initialize(weights, biases)

        for i in range(len(weights)):
            self.v_w[i] = self.momentum * self.v_w[i] - self.learning_rate * gradients_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.learning_rate * gradients_b[i]

            weights[i] += self.v_w[i]
            biases[i] += self.v_b[i]

        return weights, biases

class NesterovAcceleratedGradient(Optimizer):
    """Nesterov Accelerated Gradient"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

    def initialize(self, weights, biases):
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.initialize(weights, biases)

        for i in range(len(weights)):
            v_prev_w = self.v_w[i].copy()
            v_prev_b = self.v_b[i].copy()

            self.v_w[i] = self.momentum * self.v_w[i] - self.learning_rate * gradients_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.learning_rate * gradients_b[i]

            # Apply NAG correction
            weights[i] += -self.momentum * v_prev_w + (1 + self.momentum) * self.v_w[i]
            biases[i] += -self.momentum * v_prev_b + (1 + self.momentum) * self.v_b[i]

        return weights, biases

class RMSProp(Optimizer):
    """RMSProp Optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None

    def initialize(self, weights, biases):
        self.cache_w = [np.zeros_like(w) for w in weights]
        self.cache_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.cache_w is None:
            self.initialize(weights, biases)

        for i in range(len(weights)):
            # Update cache with squared gradients
            self.cache_w[i] = self.decay_rate * self.cache_w[i] + (1 - self.decay_rate) * (gradients_w[i] ** 2)
            self.cache_b[i] = self.decay_rate * self.cache_b[i] + (1 - self.decay_rate) * (gradients_b[i] ** 2)

            # Update parameters
            weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)

        return weights, biases

class Adam(Optimizer):
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def initialize(self, weights, biases):
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.t = 0

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.m_w is None:
            self.initialize(weights, biases)

        self.t += 1

        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]

            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return weights, biases

class NAdam(Optimizer):
    """NAdam Optimizer (Nesterov-accelerated Adam)"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def initialize(self, weights, biases):
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.t = 0

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.m_w is None:
            self.initialize(weights, biases)

        self.t += 1

        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]

            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Calculate the NAdam update (incorporating Nesterov momentum)
            m_nesterov_w = self.beta1 * m_hat_w + (1 - self.beta1) * gradients_w[i] / (1 - self.beta1 ** self.t)
            m_nesterov_b = self.beta1 * m_hat_b + (1 - self.beta1) * gradients_b[i] / (1 - self.beta1 ** self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_nesterov_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i] -= self.learning_rate * m_nesterov_b / (np.sqrt(v_hat_b) + self.epsilon)

        return weights, biases

class FeedForwardNeuralNetwork:
    def __init__(self, layer_sizes, optimizer='sgd', hidden_activation='sigmoid', init_method='random',epochs=32,batch_size=32, **optimizer_params):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.hidden_activation = hidden_activation
        self.epochs=epochs
        self.batch_size=batch_size

        # Print layer sizes for debugging
        print(f"Initializing network with layer sizes: {layer_sizes}")
        print(f"Using activation function: {hidden_activation}")
        print(f"Using initialization method: {init_method}")

        for i in range(len(layer_sizes) - 1):
            # Initialize weights based on chosen method
            if init_method.lower() == 'xavier':
                # Xavier/Glorot initialization: scale by sqrt(2 / (n_in + n_out))
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
                weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            else:  # Default to simple random initialization
                weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01

            biases = np.zeros(layer_sizes[i + 1])

            print(f"Layer {i}: Weight shape = {weights.shape}, Bias shape = {biases.shape}")

            self.weights.append(weights)
            self.biases.append(biases)

        # Set up the optimizer
        self.setup_optimizer(optimizer, **optimizer_params)

    def setup_optimizer(self, optimizer='sgd', **optimizer_params):
        """Configure the optimizer based on the provided name and parameters"""
        optimizer_map = {
            'sgd': SGD,
            'momentum': Momentum,
            'nag': NesterovAcceleratedGradient,
            'rmsprop': RMSProp,
            'adam': Adam,
            'nadam': NAdam
        }

        if optimizer.lower() in optimizer_map:
            self.optimizer = optimizer_map[optimizer.lower()](**optimizer_params)
        else:
            print(f"Warning: Unknown optimizer '{optimizer}'. Defaulting to SGD.")
            self.optimizer = SGD(**optimizer_params)

        # Initialize optimizer state
        self.optimizer.initialize(self.weights, self.biases)

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
                self.h_values.append(self.activations(self.hidden_activation, self.a_values[i]))

        return self.h_values[-1]  # Return the output

    def back_pass(self, x, y):
        """
        Computes gradients for backpropagation but doesn't update weights directly.
        Returns gradients for the optimizer to use.
        """
        m = x.shape[0]
        y_hat = self.h_values[-1]

        delta = y_hat - y

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
            current_delta = np.dot(current_delta, self.weights[i + 1].T) * self.activation_derivative(self.hidden_activation, self.h_values[i + 1])
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
        """Update weights using the configured optimizer"""
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, gradients_w, gradients_b
        )

    def activations(self, type, x):
        if type == 'sigmoid':
            # Clip x to avoid overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif type == 'relu':
            return np.maximum(0, x)
        elif type == 'tanh':
            # Clip x to avoid overflow
            x = np.clip(x, -500, 500)
            return np.tanh(x)
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
            return 1.0 * (x > 0)
        elif type == 'tanh':
            # tanh derivative is (1 - tanh²(x))
            return 1 - x**2
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

    def train(self, x, y, epochs, batch_size, learning_rate=None, optimizer=None, **optimizer_params):
        """
        Train the neural network using the configured optimizer.
        Allows for optimizer changes during training if needed.
        """
        # Update optimizer if a new one is provided
        if optimizer is not None:
            self.setup_optimizer(optimizer, **optimizer_params)

        # Update learning rate if provided
        if learning_rate is not None:
            self.optimizer.learning_rate = learning_rate

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

        # Print optimizer info
        print(f"Training with optimizer: {type(self.optimizer).__name__}")

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

                # Update weights using the optimizer
                self.update_weights(gradients_w, gradients_b)

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

