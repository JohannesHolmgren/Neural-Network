import numpy as np

from Perceptron.utils import sign


class Function:

    @classmethod
    def forward(cls, x): ...

    @classmethod
    def backwards(cls, x): ...

    @classmethod
    def evaluate(cls, output, target): ...


class Tanh(Function):

    @classmethod
    def forward(cls, local_field):
        return np.tanh(local_field)
    
    @classmethod
    def backwards(cls, local_field):
        return 1 - np.tanh(local_field)**2
    
    @classmethod
    def evaluate(cls, output, target):
        if output.shape != target.shape:
            raise ValueError('output and target must have the same shape ({output.shape} vs {target.shape})')
        error_sum = 0
        for i in range(output.shape[0]):
            error_sum += abs(sign(output[i]) - target[i])
        return 1 - error_sum / (2 * output.shape[0])
    

class Softmax(Function):

    @classmethod
    def forward(cls, local_field):
        return (np.exp(local_field)) / sum(np.exp(local_field))
    
    @classmethod
    def backwards(cls, local_field):
        return 1
    
    @classmethod
    def evaluate(cls, output, target):
        ...
        # check if the index of the maximum value of the output 
        # corresponds to the index of the onehot-econded target
        if output.shape != target.shape:
            raise ValueError('output and target must have the same shape ({output.shape} vs {target.shape})')
        validation_score = 0
        for i in range(output.shape[0]):
            output_index = np.argmax(output[i,:])
            target_index = np.argmax(target[i,:])
            if output_index == target_index:
                validation_score += 1
        
        return validation_score / output.shape[0]
        
