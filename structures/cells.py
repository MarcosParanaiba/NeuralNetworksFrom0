"""The basic Neural Cells"""
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
import tools.functions as f

class NeuralCell(ABC):
    """An Abstract Cell"""
    @abstractmethod
    def feedforward(self, inputs: np.ndarray):
        """Compute de feedfoward of the cell.
        
        By using linear transformation with the weights and bias of the
        cell returns a output.

        Parameters:
        -----------
        inputs : np.ndarray
            A 1-dimensional array that will be processed by the cell.
        
        Returns:
        --------
        float
            The output value of the transformations.
        """
    @abstractmethod
    def backwardpass(self, error: float, learning_rate: float):
        """Compute the backwardpass of the cell.

        By using a partial derivate to get the error referent of each
        attribute of the cell, outputs a error refering to it's inputs.

        Parameters:
        -----------
        error : float
            The error obtained by derivation of the next cells.
        learning_rate : float
            The impact of givin error in the matter of the learning.
        
        Returns:
        --------
        float
            The error obtained by the derivation of the cell.
        """

class DenseCell(NeuralCell):
    """An Dense Cell"""
    def __init__(self, input_size: int, function: Type[f.Function]):
        self._weights = np.random.uniform(-np.sqrt(6 / input_size),
                                          np.sqrt(6 / input_size),
                                          (input_size, 1))
        self._bias = np.zeros(1)
        self.inputs = np.empty((1, input_size))
        self.z = np.empty(1)
        self.function = function
        self.output = np.empty(1)

    def feedforward(self, inputs: np.ndarray):
        self.inputs = inputs

        self.z = np.dot(self.inputs, self._weights) + self._bias
        self.output = self.function.f(self.z)

        return self.output.item()

    def backwardpass(self, error: float, learning_rate: float):
        dbias = error * self.function.d(self.z)
        dweights = np.dot(self.inputs.reshape(-1, 1), dbias)
        dinput = np.dot(dbias, self._weights.T)

        self.weights += dweights * learning_rate
        self.bias += np.sum(dbias * learning_rate)

        return dinput

    @property
    def weights(self):
        """Get weights."""
        return self._weights
    @weights.setter
    def weights(self, value):
        """Set weights."""
        if self._weights.shape != value.shape:
            raise AttributeError("Wrong Shape")
        self._weights = value
    @property
    def bias(self):
        """Get bias."""
        return self._bias
    @bias.setter
    def bias(self, value):
        """Set bias."""
        if self._bias.shape != value.shape:
            raise AttributeError("Wrong Shape")
        self._bias = value

class RecurrentCell(DenseCell):
    def __init__(self, input_size, function = f.Tanh):
        super().__init__(input_size, function)
        self._hweights = np.random.uniform(-np.sqrt(6 / input_size),
                                          np.sqrt(6 / input_size), 1)
        self.h = self.output

    def feedforward(self, inputs):
        self.inputs = inputs
        self.h = self.output

        hz = self.h * self._hweights
        self.z = np.dot(self.inputs, self._weights) + self._bias + hz
        self.output = self.function.f(self.z)

        return self.output.item()

    def backwardpass(self, error, learning_rate):
        dbias = error * self.function.d(self.z)
        dweights = np.dot(self.inputs.reshape(-1, 1), dbias)
        dhweights = self.h * dbias
        dinput = np.dot(dbias, self._weights.T)

        self._weights += dweights * learning_rate
        self._hweights += sum(dhweights * learning_rate)
        self.bias += np.sum(dbias * learning_rate)

        return dinput
