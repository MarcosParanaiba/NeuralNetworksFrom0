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
    """An Dense Cell
    
    By basic linear transformations, process data and correct their

    Attributes:
    -----------
        weights (np.ndarray): The weight matrix.

        bias (np.ndarray): The bias term.

        inputs (np.ndarray): The input values for the neuron during
        feedforward.

        z (np.ndarray): The linear combination of inputs and weights
        plus bias.

        function (Type[f.Function]): The activation function applied to
        the linear combination.

        output (np.ndarray): The output of the neuron after applying the
        linear transformations.
    """
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
    """An Recurrent Cell
    

    Attributes:
    -----------
        weights (np.ndarray): The weight matrix.

        bias (np.ndarray): The bias term.

        inputs (np.ndarray): The input values for the neuron during
        feedforward.

        z (np.ndarray): The linear combination of inputs and weights
        plus bias.

        function (Type[f.Function]): The activation function applied to
        the linear combination.

        output (np.ndarray): The output of the neuron after applying the
        activation function.
    """
    def __init__(self, input_size, function=f.TanH):
        super().__init__(input_size + 1, function)
    def feedforward(self, inputs):
        self.inputs = np.append(inputs, self.output)

        return super().feedforward(self.inputs)

    def backwardpass(self, error, learning_rate):
        return super().backwardpass(error, learning_rate)[:-1]

class ConvolutionalCell(NeuralCell):
    """An Recurrent Cell
    

    Attributes:
    -----------
        kernele (np.ndarray): The kernel matrix.

        bias (np.ndarray): The bias term.

        inputs (np.ndarray): The input values for the neuron during
        feedforward.

        z (np.ndarray): The sum of the channels' convolutions.

        function (Type[f.Function]): The activation function applied to
        the linear combination.

        output (np.ndarray): The output of the neuron after applying the
        activation function.
    """
    def __init__(self, input_size: int, kernel_size: int,
                 input_channels: int, function: Type[f.Function]):
        self._kernel = np.random.uniform(-np.sqrt(6 / input_size),
                                          np.sqrt(6 / input_size),
                                          (kernel_size, kernel_size))
        self._bias = np.zeros(1)
        self.inputs = np.empty((input_size, input_size))
        self.z = np.empty((input_size - kernel_size + 1,
                           input_size - kernel_size + 1),
                           input_channels)
        self.function = function
        self.output = np.empty((input_size - kernel_size + 1,
                                input_size - kernel_size + 1))
    def convolution(self, x: np.ndarray, k: np.ndarray,
                    padding: int = 0):
        """The convolution function

        Parameters:
        -----------
            x (np.ndarray): The matrix to be convolutioned.
            k (np.ndarray): The kernel that will convolutionize.
            padding (int): How much will the x will be padded.
                If 'full' the convolution will be full

        """
        x_size = x.shape[0]
        k_size = k.shape[0]
        reduction = k_size - 1

        if padding == 'full':
            x = np.pad(x, reduction, 'constant',
                    constant_values = 0)

        if padding > 0:
            x = np.pad(x, min(reduction, padding), 'constant',
                    constant_values = 0)

        return np.array([[np.sum(x[r : r + k_size, c : c + k_size] * k
                 for c in range(x_size - reduction))]
                for r in range(x_size-reduction)])

    def feedforward(self, inputs):
        self.inputs = inputs

        self.z = np.sum(self.convolution(inputs, self.kernel), axis = 2)

        self.output = self.function(self.z)

        return self.output

    def backwardpass(self, error, learning_rate):
        dbias  = error * self.function.d(self.z)
        dkernel = self.convolution(self.inputs, dbias)
        dinput = self.convolution(np.rot90(self._kernel, 2), dbias)

        self.bias += dbias * learning_rate
        self.kernel += dkernel * learning_rate

        return dinput

    @property
    def kernel(self):
        """Get weights."""
        return self._kernel
    @kernel.setter
    def weights(self, value):
        """Set weights."""
        if self._kernel.shape != value.shape:
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
