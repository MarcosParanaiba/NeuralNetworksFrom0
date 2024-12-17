"""The basic Neural Layers"""
from typing import Type
import numpy as np
import tools.functions as f
import structures.cells as c

class NeuralLayer():
    """An Neural Layer
    
    Attributes:
    -----------
        cells (list[Type[c.NeuralCell]]): A list of all the cells in the
        layer

        weights (np.ndarray): The set of all weight matrix.

        bias (np.ndarray): The set af bias term.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.
    """
    def __init__(self, input_size: int, *args: Type[c.NeuralCell]):
        size = len(args)
        self.cells = list(args)
        self.inputs = np.empty((1, input_size))
        self.output = np.empty((1, size))

    def feedforward(self, inputs):
        """Compute de feedfoward of the layer.
        
        Gethers all the feedfowards given by it's cells.

        Parameters:
        -----------
        inputs : np.ndarray
            A 1-dimensional array that will be processed by the layer.
        
        Returns:
        --------
        np.ndarray
            The output value of the transformations.
        """
        self.inputs = inputs
        self.output = np.array([cell.feedforward(inputs)
                         for cell in self.cells])
        return self.output

    def backwardpass(self, error, learning_rate):
        """Compute the backwardpass of the layer.

        By using a partial derivate to get the error referent of each
        neural cell and their respectively backpropagation, outputs a
        error refering to it's inputs.

        Parameters:
        -----------
        error : np.ndarray
            The error obtained by derivation of the next layers.
        learning_rate : float
            The impact of givin error in the matter of the learning.
        
        Returns:
        --------
        np.ndarray
            The error obtained by the derivation of the layer.
        """
        error = np.array([cell.backwardpass(error[i].reshape(1,1),
                                            learning_rate)
                          for (i, cell) in enumerate(self.cells)])
        return np.sum(error, axis=0).squeeze()

    @property
    def weights(self):
        """Get weights."""
        return np.array([cell.weights for cell in self.cells])
    @weights.setter
    def weights(self, value):
        """Set weights."""
        for (i, v) in enumerate(value):
            self.cells[i].weights = v

    @property
    def biases(self):
        """Get biases."""
        return np.array([cell.bias for cell in self.cells])
    @biases.setter
    def biases(self, value):
        """Set bias."""
        for (i, v) in enumerate(value):
            self.cells[i].bias = v

class DenseLayer(NeuralLayer):
    """An Neural Layer
    
    Attributes:
    -----------
        cells (list[Type[c.DenseCell]]): A list of all the cells in the
        layer

        weights (np.ndarray): The set of all weight matrix.

        bias (np.ndarray): The set af bias term.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.
    """
    def __init__(self, input_size: int, size: int,
                 function: Type[f.Function]):
        super().__init__([c.DenseCell(input_size, function)
                      for _ in range(size)])

class RecurrentLayer(DenseLayer):
    """An Neural Layer
    
    Attributes:
    -----------
        cells (list[Type[c.RecurrentCell]]): A list of all the cells in
        the layer

        weights (np.ndarray): The set of all weight matrix.

        bias (np.ndarray): The set af bias term.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.
    """
    def __init__(self, input_size, size, function = f.TanH):
        super().__init__(input_size, size, function)
        self.cells = [c.RecurrentCell(input_size, function)
                      for _ in range(size)]
