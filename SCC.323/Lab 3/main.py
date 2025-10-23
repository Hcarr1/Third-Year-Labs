import MLP
import numpy as np

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [1]])

    mlp = MLP.MLP(input_features=2, hidden_nodes=4, output_nodes=1, learning_rate=0.1, max_iter=10000)
    mlp.train(X, y)
    print(mlp.predict([1,1]))