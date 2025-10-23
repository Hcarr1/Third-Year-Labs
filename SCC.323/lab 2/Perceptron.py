import activations
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, max_iter=1000):
        """
        Initialize the perceptron with given input size and learning rate.
        :param input_size: The number of input features.
        :param learning_rate:
        """
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Train the perceptron using the provided training data.
        :param X: List of input feature vectors.
        :param y: List of target labels.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        for _ in range(self.max_iter):
            errors = 0
            for Xi, yi in zip(X, y):
                v = np.dot(self.weights, Xi) + self.bias
                output = activations.Heaviside(v)
                delta = yi - output
                if delta != 0:
                    self.weights += self.lr * delta * Xi
                    self.bias += self.lr * delta
                    errors += 1
            if errors == 0:
                break

    def predict(self, X) -> list:
        """
        Predict the output for the given input data.
        :param X: List of input feature vectors.
        :return: List of predicted labels.
        """
        X = np.asarray(X)

        nets = X.dot(self.weights) + self.bias
        predictions = [activations.Heaviside(i) for i in nets]

        return predictions
