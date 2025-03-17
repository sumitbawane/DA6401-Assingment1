
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
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]

            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)

            
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)

           
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            
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
            
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]

            
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)

          
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)

           
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            
            m_nesterov_w = self.beta1 * m_hat_w + (1 - self.beta1) * gradients_w[i] / (1 - self.beta1 ** self.t)
            m_nesterov_b = self.beta1 * m_hat_b + (1 - self.beta1) * gradients_b[i] / (1 - self.beta1 ** self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_nesterov_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i] -= self.learning_rate * m_nesterov_b / (np.sqrt(v_hat_b) + self.epsilon)

        return weights, biases