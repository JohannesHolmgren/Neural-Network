import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" 
import functions
from functions import Tanh
from layer import Layer
"""


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


def sign(x):
    return -1 if x < 0 else 1


# =============================


class Perceptron:
    ''' A class representing a perceptron with hidden layers
        using backpropagation for updating the weights '''

    def __init__(self, eta, alpha=0.9):
        self.layers = []
        self.weights = []
        self.dweights = []
        self.prev_dweights = []

        self.eta = eta
        self.alpha = alpha

        # For plotting etc.
        self.training_scores = []
        self.validation_scores = []
        # self.best_weights = [self.weights[0], self.weights[-1]]
        # self.best_thresholds = [self.layers[1].thresholds, self.layers[-1].thresholds]
        self.best_validation_score = 100
        # self.early_stopping_counter = 0
        # self.early_stopping_threshold = 5

    def add(self, layer):
        self.layers.append(layer)

    def load_parameters(self, dir):
        w1 = np.genfromtxt(dir + '/w1.csv', delimiter=",")
        w2 = np.genfromtxt(dir + '/w2.csv', delimiter=",")
        t1 = np.genfromtxt(dir + '/t1.csv', delimiter=",")
        t2 = np.genfromtxt(dir + '/t2.csv', delimiter=",")
        self.layers[1].thresholds = t1
        self.layers[2].thresholds = t2
        self.weights.append(w1)
        self.weights.append(w2)

    def init_weights(self):
        ''' Init weights from a normal distribution with
            mean 0 and variance 1/n_input '''
        mean = 0

        for i in range(1, len(self.layers)):
            std = math.sqrt(1 / self.layers[i-1].size)  # standard deviation 1 / m where m is size of previous layer
            mean = 0
            size=(self.layers[i].size, self.layers[i-1].size)
            weights = np.random.default_rng().normal(mean, std, size=size)
            self.weights.append(weights)
            self.dweights.append(np.zeros(size))
            self.prev_dweights.append(np.zeros(size))

    def forward(self):
        ''' Forward iteration '''
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].nodes, self.weights[i-1])
        return self.layers[-1].nodes

    def backwards(self, target):
        ''' backwards iteration '''

        # Calculate error, dweights and dthreshold for output layer and weights incoming to output layer
        output_error = (target - self.layers[-1].nodes)
        self.layers[-1].errors = output_error * self.layers[-1].activation_function.backwards(self.layers[-1].local_field)
        self.layers[-1].update_dthresholds()
        self.dweights[-1] =  self.layers[-2].nodes * self.layers[-1].errors.reshape(-1, 1)

        # Calculate error, dweights and dthreshold for hidden layers and weights incoming to hidden layers
        for i in range(len(self.layers)-2, 0, -1):
            self.layers[i].backwards(self.layers[i+1].errors, self.weights[i])
            self.dweights[i-1] = self.layers[i-1].nodes * self.layers[i].errors.reshape(-1, 1)
        
    def update(self):
        ''' Update all weights and thresholds '''

        # Update weights to output layer
        self.prev_dweights[-1] = self.eta * self.dweights[-1] + self.alpha * self.prev_dweights[-1]
        self.weights[-1] = np.add(self.weights[-1], self.prev_dweights[-1])
        self.dweights[-1] = np.zeros_like(self.dweights[-1])

        # Output output thresholds
        output_dthreshold = (self.eta * self.layers[-1].dthreshold)
        self.layers[-1].thresholds -= output_dthreshold
        self.layers[-1].dthreshold = np.zeros_like(self.layers[-1].dthreshold)

        # Hidden layers
        for i in range(len(self.layers)-2, 0, -1):

            # Update weights
            self.prev_dweights[i-1] = self.eta * self.dweights[i-1] + self.alpha * self.prev_dweights[i-1]
            self.weights[i-1] = np.add(self.weights[i-1], self.prev_dweights[i-1])
            self.dweights[i-1] = np.zeros_like(self.dweights[i-1])

            # Update thresholds
            dthreshold = (self.eta * self.layers[i].dthreshold)
            self.layers[i].thresholds -= dthreshold
            self.layers[i].dthreshold = np.zeros_like(self.layers[i].dthreshold)

    def predict(self, input_vector):
        self.layers[0].nodes = input_vector.copy()
        predicted = self.forward()
        return predicted

    def train(self, training_data, validation_data, epochs=10, statistics=True, training_method='batch'):


        output_values = []
        targets = []
        for row in training_data:
            input_data, target = self.split_data(row)
            self.layers[0].nodes = input_data.copy()
            targets.append(target)
            output_values.append(self.forward())
        print(self.layers[-1].activation_function.evaluate(np.array(output_values), np.array(targets)))

        for i in range(epochs):
            if training_method == 'batch':
                self.batch(training_data, batch_size=8)
            elif training_method == 'sequential':
                self.sequential(training_data)
            elif training_method == 'stochastic_sequential':
                self.stochastic_sequential(training_data)
            elif training_method == 'stochastic_batch':
                self.stochastic_batch(training_data, batch_size=8)

            output_values = []
            targets = []
            for row in training_data:
                input_data, target = self.split_data(row)
                self.layers[0].nodes = input_data.copy()
                targets.append(target)
                output_values.append(self.forward())
            print(self.layers[-1].activation_function.evaluate(np.array(output_values), np.array(targets)))
            

            # Statistics
            if statistics:
                validation_score = self.evaluate(validation_data)
                if i % 10 == 0:
                    print(f'Epoch {i}: {validation_score}')
                self.validation_scores.append(validation_score)
                if validation_score < self.best_validation_score:
                    self.best_weights = [
                        self.weights[0].copy(),
                        self.weights[-1].copy()]
                    self.best_thresholds = [
                        self.layers[1].thresholds.copy(),
                        self.layers[-1].thresholds.copy()]
                    self.best_validation_score = validation_score

    def sequential(self, training_data):
        ''' train the network sequentially (update weights
            and threshold after each training vector). '''
        for training_vector in training_data:
            input_data, target = self.split_data(training_vector)
            self.layers[0].nodes = input_data.copy()
            self.forward()
            self.backwards(target)
            # Update after each processed input
            self.update()

    def stochastic_sequential(self, training_data):
        ''' train the network sequentially but choose a random
            training vector each iteration. '''
        number_of_rows = len(training_data)
        indices = np.random.choice(number_of_rows, number_of_rows)
        self.sequential(training_data[indices, :])

    # TODO USE BATCH SIZE
    def batch(self, training_data, batch_size):
        ''' train the network as a batch of entire dataset
            and update weights and thresholds after the batch. '''
        for training_vector in training_data:
            input_data, target = self.split_data(training_vector)
            self.layers[0].nodes = input_data.copy()
            self.forward()
            self.backwards(target)
        # Update after entire batch is iterated
        self.update()

    def stochastic_batch(self, training_data, batch_size=512):
        ''' traing the network as a batch of a portion of the dataset.
            The batch is chosen randomly and weights and thresholds are
            updated after the batch. '''
        temp = training_data.copy()
        np.random.shuffle(temp)
        for batch in np.array_split(temp, len(temp)/batch_size):
            self.batch(batch)

    def split_data(self, data_vector):
        ''' Split the vector into input and target '''
        input_size = self.layers[0].size
        input_data = data_vector[0:input_size]
        target = data_vector[input_size:]
        return input_data, target

    def evaluate(self, validation_data):
        error_sum = 0
        for validation_vector in validation_data:
            input_data, target = self.split_data(validation_vector)
            output = self.predict(input_data)
            error_sum += abs(sign(output) - target)
        return error_sum / (2 * validation_data.shape[0])
    
    def plot_epoch(self):
        plt.plot(self.validation_scores, 'r')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Error')
        plt.xticks(range(0, EPOCHS, 10))
        plt.show()

    def save_parameters(self):
        w1 = pd.DataFrame(self.best_weights[0])
        w2 = pd.DataFrame(self.best_weights[1].transpose())
        t1 = pd.DataFrame(self.best_thresholds[0])
        t2 = pd.DataFrame(self.best_thresholds[1])
        w1.to_csv('saves/w1.csv', index=False, header=False)
        w2.to_csv('saves/w2.csv', index=False, header=False)
        t1.to_csv('saves/t1.csv', index=False, header=False)
        t2.to_csv('saves/t2.csv', index=False, header=False)


if __name__ == '__main__':


    if False:

        p = Perceptron(0.05)
        p.add(Layer(2))
        p.add(Layer(12))
        p.add(Layer(1))
        p.init_weights()

        training_data, validation_data = read_data(
            'training_set.csv', 'validation_set.csv')
        
        p.train(training_data, validation_data, epochs=100, training_method='stochastic_batch')


    if True:
        ETA = 0.001
        INPUT_SIZE = 2
        HIDDEN_SIZE = 50
        OUTPUT_SIZE = 1
        EPOCHS = 100

        training_data, validation_data = read_data(
            'training_set.csv', 'validation_set.csv')

        p = Perceptron(ETA)
        p.add(Layer(INPUT_SIZE, Tanh))
        p.add(Layer(HIDDEN_SIZE, Tanh))
        p.add(Layer(OUTPUT_SIZE, Tanh))
        p.init_weights()

        p.train(training_data, validation_data, epochs=EPOCHS, training_method='stochastic_batch')

        vs = p.evaluate(validation_data)
        print(f'Final validation score: {vs}, best validation score: {p.best_validation_score}')

        # p.save_parameters()

        # p.plot_epoch()
