import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Perceptron.functions import Tanh
from Perceptron.layer import Layer


class Perceptron:
    ''' A class representing a perceptron with hidden layers
        using backpropagation for updating the weights '''

    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.weights = []
        self.dweights = []
        self.prev_dweights = []

        self.learning_rate = learning_rate
        self.alpha = 0.9

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
        self.prev_dweights[-1] = self.learning_rate * self.dweights[-1] + self.alpha * self.prev_dweights[-1]
        self.weights[-1] = np.add(self.weights[-1], self.prev_dweights[-1])
        self.dweights[-1] = np.zeros_like(self.dweights[-1])

        # Output output thresholds
        output_dthreshold = (self.learning_rate * self.layers[-1].dthreshold)
        self.layers[-1].thresholds -= output_dthreshold
        self.layers[-1].dthreshold = np.zeros_like(self.layers[-1].dthreshold)

        # Hidden layers
        for i in range(len(self.layers)-2, 0, -1):

            # Update weights
            self.prev_dweights[i-1] = self.learning_rate * self.dweights[i-1] + self.alpha * self.prev_dweights[i-1]
            self.weights[i-1] = np.add(self.weights[i-1], self.prev_dweights[i-1])
            self.dweights[i-1] = np.zeros_like(self.dweights[i-1])

            # Update thresholds
            dthreshold = (self.learning_rate * self.layers[i].dthreshold)
            self.layers[i].thresholds -= dthreshold
            self.layers[i].dthreshold = np.zeros_like(self.layers[i].dthreshold)

    def predict(self, input_vector):
        self.layers[0].nodes = input_vector.copy()
        predicted = self.forward()
        return predicted

    def train(self, training_data, validation_data=None, epochs=10, training_method='batch', batch_size=8):
        for i in range(epochs):
            if training_method == 'sequential':
                self.sequential(training_data)
            elif training_method == 'stochastic_sequential':
                self.stochastic_sequential(training_data)
            elif training_method == 'batch':
                self.batch(training_data, batch_size)
            elif training_method == 'stochastic_batch':
                self.stochastic_batch(training_data, batch_size)

            if validation_data is not None:
                validation_score = self.evaluate(validation_data)
                print(f'Epoch {i+1} score: {validation_score}')

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
        for batch in np.array_split(training_data, len(training_data)/batch_size):
            for training_vector in batch:
                input_data, target = self.split_data(training_vector)
                self.layers[0].nodes = input_data.copy()
                self.forward()
                self.backwards(target)
            # Update after entire batch is iterated
            self.update()

    def stochastic_batch(self, training_data, batch_size):
        ''' traing the network as a batch of a portion of the dataset.
            The batch is chosen randomly and weights and thresholds are
            updated after the batch. '''
        shuffled_data = training_data.copy()
        np.random.shuffle(shuffled_data)
        self.batch(shuffled_data, batch_size)

    def split_data(self, data_vector):
        ''' Split the vector into input and target 
            using the size of the input layer. '''
        input_size = self.layers[0].size
        input_data = data_vector[0:input_size]
        target = data_vector[input_size:]
        return input_data, target

    def evaluate(self, validation_data):
        ''' Evaluate the model on the validation data by feeding 
            it to the model and then comparing the output with the targets,
            using the evaluate method on the activation function. '''
        size = (validation_data.shape[0], self.layers[-1].nodes.shape[0])
        output_values = np.zeros(size)
        targets = np.zeros(size)
        for i, row in enumerate(validation_data):
            input_vector, target = self.split_data(row)
            targets[i,:] = target
            output = self.predict(input_vector)
            output_values[i,:] = output
        return self.layers[-1].activation_function.evaluate(output_values, targets)
    

# TODO
# Remove the need for calling init_weights 
# Explain learning_rate and alpha and give both a standard value
# Look if anything more can be extracted to other classes
# Make sure variable names do not differ
# Good comments for everything
# Make into a package
# Look at imports to make them independent of where program is run
