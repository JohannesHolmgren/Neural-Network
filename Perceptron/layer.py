import numpy as np

from Perceptron import functions


class Layer:
    ''' A class representing a layer in an neural network '''

    def __init__(self, size, activation_function):
        # Metadata
        self.size = size

        # Values and thresholds
        self.nodes = np.zeros(size)
        self.thresholds = np.zeros(size)

        # For backpropagation
        self.local_field = np.zeros(size)
        self.dthreshold = np.zeros(size)
        self.errors = np.zeros(size)

        # Activation function
        if not issubclass(activation_function, functions.Function):
            raise TypeError(f"activation_function must be of instance \'{functions.Function.__name__}\'")
        self.activation_function = activation_function

    def forward(self, input_nodes, incoming_weights):
        self.local_field = incoming_weights.dot(input_nodes) - self.thresholds
        self.nodes = self.activation_function.forward(self.local_field)

    def backwards(self, future_errors, outgoing_weights):
        self.update_errors(future_errors, outgoing_weights)
        self.update_dthresholds()
    
    def update_errors(self, future_errors, outgoing_weights):
        self.errors = (future_errors.dot(outgoing_weights)) * self.activation_function.backwards(self.local_field)

    def update_dthresholds(self):
        self.dthreshold += self.errors

    
class ConvolutionLayer:


    def __init__(self, size, stride, padding):
        self.size = size
        self.stride = stride
        self.padding = padding
        self.kernel = np.array(size)

    def forward(self, image):
        ''' Image is a 2d-matrix '''
        ...
        # Add padding
        # iterate from upper left to bottom right with stride size