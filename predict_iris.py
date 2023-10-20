import numpy as np

from Perceptron.perceptron import Perceptron
from Perceptron.layer import Layer
from Perceptron.functions import Softmax, Tanh

def load_iris_data(data_path, label_path):
    iris_data = np.genfromtxt(data_path, delimiter=',')
    max_vector = iris_data.max(axis=0)
    iris_data = iris_data / max_vector
    iris_labels = np.genfromtxt(label_path, delimiter=',')
    return iris_data, iris_labels

def onehot(data, n_labels):
    onehot_data = np.zeros([data.shape[0], n_labels])
    for i, label in enumerate(data):
        index = int(label)
        onehot_data[i, index] = 1
    return onehot_data



if __name__ == '__main__':
    
    # Load data and convert lables to onehot encoding
    iris_data, iris_labels = load_iris_data('data/iris-data.csv', 'data/iris-labels.csv')
    iris_labels = onehot(iris_labels, 3)

    data = np.append(iris_data, iris_labels, axis=1)

    # Build neural network
    network = Perceptron(0.001)
    network.add(Layer(4, Tanh))     # Input layer
    network.add(Layer(10, Tanh))    # Hidden layer
    network.add(Layer(3, Softmax))  # Output layer
    network.init_weights()

    network.train(data, data, epochs=100, statistics=False, training_method='sequential')