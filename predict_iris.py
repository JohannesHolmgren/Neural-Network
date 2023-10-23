import numpy as np

from Perceptron.functions import Softmax, Tanh
from Perceptron.layer import Layer
from Perceptron.perceptron import Perceptron


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

def split(data, frac=0.8):
    shuffled_data = data.copy()
    np.random.shuffle(shuffled_data)
    split_index = int(shuffled_data.shape[0]*frac)
    return shuffled_data[:split_index], shuffled_data[split_index:]

if __name__ == '__main__':
    
    # Load data and convert lables to onehot encoding
    iris_data, iris_labels = load_iris_data('data/iris-data.csv', 'data/iris-labels.csv')
    iris_labels = onehot(iris_labels, 3)

    data = np.append(iris_data, iris_labels, axis=1)
    training_data, validation_data = split(data, 0.8)

    # Build neural network
    network = Perceptron(learning_rate=0.001)
    network.add(Layer(4, Tanh))     # Input layer
    network.add(Layer(10, Tanh))    # Hidden layer
    network.add(Layer(10, Tanh))    # Hidden layer
    network.add(Layer(3, Softmax))  # Output layer
    network.init_weights()

    network.train(training_data, validation_data, epochs=50, training_method='sequential')