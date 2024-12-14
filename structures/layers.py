"""The basic Neural Layers"""
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
import tools.functions as f
import structures.cells as c

class NeuralLayer(ABC):
    """An Abstract Layer"""
    @abstractmethod
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
    @abstractmethod
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

class DenseLayer(NeuralLayer):
    """An Dense Layer"""
    def __init__(self, input_size: int, size: int,
                 function: Type[f.Function]):
        self.cells = [c.DenseCell(input_size, function)
                      for _ in range(size)]
        self.inputs = np.empty((1, input_size))
        self.output = np.empty((1, size))

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = np.array([cell.feedforward(inputs)
                         for cell in self.cells])
        return self.output

    def backwardpass(self, error, learning_rate):
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
