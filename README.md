# Neural-Network
This is a small neural network package using backpropagation for updating weights and thresholds. It roots from a course in Artificial Neural Networks at Chalmers University of Technology.

## Features
- Supports an arbitrary number of layers, all with arbitrary sizes.
- Implemented activation functions:
  - Tanh - for values in the range $[-1, 1]$.
  - Softmax - for output layer to classify different inputs using one-hot encoding.
- Adaptive weight initialization using connected layer size.
- Batch and sequential training, both stochastic and non-stochastic.
- Adaptive learning rate to avoid local minima.
