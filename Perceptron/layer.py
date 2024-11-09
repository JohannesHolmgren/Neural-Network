# from Perceptron import functions
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


    def __init__(self, size):
        self.size = size    # Kernel size
        self.stride = (1, 1)
        self.padding = (0, 0, 0, 0)  # Top Right Bottom Left
        self.kernel = np.ones(size)

    def pad(self, image):
        padded_image = image.copy()
        # Top
        top_padding = np.zeros((self.padding[0], padded_image.shape[1]))
        padded_image = np.vstack((top_padding, padded_image))
        # Right
        right_padding = np.zeros((padded_image.shape[0], self.padding[1]))
        padded_image = np.hstack((padded_image, right_padding))
        # Bottom
        bot_padding = np.zeros((self.padding[2], padded_image.shape[1]))
        padded_image = np.vstack((padded_image, bot_padding))
        # Left
        left_padding = np.zeros((padded_image.shape[0], self.padding[3]))
        padded_image = np.hstack((left_padding, padded_image))
        # iterate from upper left to bottom right with stride size
        return padded_image

    def forward(self, image):
        ''' Image is a 2d-matrix '''
        padded_image = self.pad(image)

        height = padded_image.shape[0] - self.size[0] + 1
        width  = padded_image.shape[1] - self.size[1] + 1
        feature_map = np.zeros([height, width])

        for j in range(0, height, self.stride[0]):
            for i in range(0, width, self.stride[1]):
                m = padded_image[j:j+self.size[0], i:i+self.size[1]]
                feature_map[j, i] = np.sum(np.multiply(self.kernel, m))
        print(feature_map)

c = ConvolutionLayer((3, 3))
c.forward(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
