import numpy as np
import activations
import LossFunctions

class MLP:
    def __init__(self, input_features, hidden_nodes, output_nodes, learning_rate, max_iter):
        # Initialize weights and biases
        self.input_features = input_features
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.max_iter = max_iter

        self.first_layer_weights = np.random.rand(self.hidden_nodes, self.input_features)
        self.second_layer_weights = np.random.rand(self.hidden_nodes, self.output_nodes)

        self.sigmoid_vectorized = np.vectorize(activations.Sigmoid)
        self.sigmoid_derivative_vectorized = np.vectorize(activations.Sigmoid_derivative)
        self.relu_vectorized = np.vectorize(activations.ReLU)
        self.relu_derivative_vectorized = np.vectorize(activations.ReLU_derivative)

    def train(self, X, y):
        for iter in range(self.max_iter):
            predictions = []
            for Xi, yi in zip(X, y):
                # Forward pass
                for x in range(self.hidden_nodes):
                    hidden_layer_input = np.array([sum(Xi * self.first_layer_weights[x])])
                # combine inputs for each hidden node
                hidden_layer_output = self.sigmoid_vectorized(hidden_layer_input)

                output_layer_input = sum(hidden_layer_output * self.second_layer_weights)
                predicted = self.relu_vectorized(output_layer_input)

                predictions.append(predicted)

                # Backward pass
                error = yi - predicted

                delta_1 = error * self.sigmoid_derivative_vectorized(output_layer_input)
                new_second_layer_weights = self.second_layer_weights + self.learning_rate * delta_1 * hidden_layer_output

                delta_2 = delta_1 * self.second_layer_weights * self.relu_derivative_vectorized(hidden_layer_input)
                new_first_layer_weights = self.first_layer_weights + self.learning_rate * delta_2 * Xi

                self.first_layer_weights = new_first_layer_weights
                self.second_layer_weights = new_second_layer_weights

            loss = LossFunctions.MSE_loss(y, predictions, len(y))
            print(f"Iteration {iter+1}/{self.max_iter}, Loss: {loss}")

    def predict(self, X):
        predictions = []
        for Xi in X:
            # Forward pass
            for x in range(self.hidden_nodes):
                hidden_layer_input = np.array([sum(Xi * self.first_layer_weights[x])])
            hidden_layer_output = self.sigmoid_vectorized(hidden_layer_input)
            output_layer_input = sum(hidden_layer_output * self.second_layer_weights)
        predicted = self.relu_vectorized(output_layer_input)
        return predicted
