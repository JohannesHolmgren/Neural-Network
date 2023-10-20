import matplotlib.pyplot as plt
import numpy as np

from Perceptron.functions import Tanh
from Perceptron.layer import Layer
from Perceptron.perceptron import Perceptron


# ===== General functions =====
def plot_data(dataset):
    targets = dataset[:, 2]
    targets = ((targets + 1) / 2).astype(int)
    colormap = np.array(['b', 'r'])
    plt.scatter(dataset[:, 0], dataset[:, 1], s=0.2, c=colormap[targets])


def read_data(training_path, validation_path):
    training_data = np.genfromtxt(training_path, delimiter=',')
    validation_data = np.genfromtxt(validation_path, delimiter=',')

    training_data, validation_data = normalize_data(
        training_data, validation_data)
    return training_data, validation_data


def normalize_data(t_data, v_data):
    # Get mean and std
    means = np.array([np.mean(t_data[:, 0]), np.mean(t_data[:, 1])])
    std = np.array([np.std(t_data[:, 0]), np.std(t_data[:, 1])])
    # Update t_data
    t_data[:, [0, 1]] = (t_data[:, [0, 1]] - means) / std
    v_data[:, [0, 1]] = (v_data[:, [0, 1]] - means) / std
    return t_data, v_data

# =============================

if __name__ == '__main__':
        ETA = 0.001
        INPUT_SIZE = 2
        HIDDEN_SIZE = 50
        OUTPUT_SIZE = 1
        EPOCHS = 100

        training_data, validation_data = read_data(
            'data/training_set.csv', 'data/validation_set.csv')

        p = Perceptron(ETA)
        p.add(Layer(INPUT_SIZE, Tanh))
        p.add(Layer(HIDDEN_SIZE, Tanh))
        p.add(Layer(OUTPUT_SIZE, Tanh))
        p.init_weights()

        p.train(training_data, validation_data, epochs=EPOCHS, training_method='batch')

        vs = p.evaluate(validation_data)
