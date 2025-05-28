import numpy as np
from sklearn.metrics import log_loss  # Importa a função de cross-entropy

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def compute_loss(self, y_true, y_pred):
        return log_loss(y_true, y_pred, labels=np.arange(y_pred.shape[1]))

    def backward(self, X, y_true):
        m = X.shape[0]
        y_one_hot = np.zeros_like(self.final_output)
        y_one_hot[np.arange(m), y_true] = 1

        error_output = (self.final_output - y_one_hot) / m
        d_weights_hidden_output = np.dot(self.hidden_output.T, error_output)
        d_bias_output = np.sum(error_output, axis=0, keepdims=True)

        error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_input)
        d_weights_input_hidden = np.dot(X.T, error_hidden)
        d_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)

        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
