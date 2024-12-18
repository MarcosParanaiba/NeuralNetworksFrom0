"""The basic Neural Layers"""
from abc import ABC, abstractmethod
from typing import Type
import numpy as np
import tools.functions as f
import structures.cells as c

class NeuralLayer(ABC):
    """An abstract Neural Layer"""
    @abstractmethod
    def feedforward(self, inputs: np.ndarray):
        """Abstract method of a feedfoward"""

    @abstractmethod
    def backwardpass(self, error: np.ndarray, learning_rate: float):
        """Abstract method of a backwardpass"""

class WildLayer(NeuralLayer):
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

class DenseLayer(WildLayer):
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

class ConvolutionalLayer(WildLayer):
    """An Canvolutional Layer
    
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
    def __init__(self, input_size: int, size: int, kernel_size: int,
                 input_channels: int, function: Type[f.Function]):
        super().__init__([c.ConvolutionalCell(input_size, kernel_size,
                                              input_channels, function)
                      for _ in range(size)])

    def feedforward(self, inputs):
        """Compute de feedfoward of the layer.
        
        Gethers all the feedfowards given by it's cells.

        Parameters:
        -----------
        inputs : np.ndarray
            A 3-dimensional array that will be processed by the layer.
        
        Returns:
        --------
        np.ndarray
            The output value of the transformations.
        """
        return super().feedforward(inputs)

    @property
    def weights(self):
        """Get weights."""
        return np.array([cell.kernel for cell in self.cells])
    @weights.setter
    def weights(self, value):
        """Set weights."""
        for (i, v) in enumerate(value):
            self.cells[i].kernel = v

    @property
    def biases(self):
        """Get biases."""
        return np.array([cell.bias for cell in self.cells])
    @biases.setter
    def biases(self, value):
        """Set bias."""
        for (i, v) in enumerate(value):
            self.cells[i].bias = v

class PoolingLayer(NeuralLayer):
    """An Layer that Pools the input by a factor
    
    Attributes:
    -----------
        pooling_size (int): The range of the pooling in the input.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.
    """
    def __init__(self, pooling_size):
        self.pooling_size = pooling_size
        self.inputs = None
        self.output = None

    @abstractmethod
    def _pool(self, window, coords):
        pass

    @abstractmethod
    def _unpool(self, dinput, error, coords):
        pass

    def feedforward(self, inputs):
        """Compute de feedfoward of the layer.
        
        Gethers all the feedfowards given by it's cells.

        Parameters:
        -----------
        inputs : np.ndarray
            A 3-dimensional array that will be processed by the layer.
        
        Returns:
        --------
        np.ndarray
            The output value of the transformations.
        """
        self.inputs = inputs
        input_size = inputs.shape[:-1]
        channels = inputs.shape[-1]
        reduction = input_size - self.pooling_size + 1

        self.output = np.empty((reduction[0], reduction[1], channels))

        for channel in range(channels):
            for raxis in range(reduction[0]):
                for caxis in range(reduction[1]):
                    window = inputs[raxis : raxis + self.pooling_size,
                                    caxis : caxis + self.pooling_size,
                                    channel]

                    self._pool(window, (raxis, caxis, channel))

        self.output = np.array(self.output)
        return self.output

    def backwardpass(self, error, _):
        """Compute the backwardpass of the layer.

        By using a partial derivate to get the error referent of each
        neural cell and their respectively backpropagation, outputs a
        error refering to it's inputs.

        Parameters:
        -----------
        error : np.ndarray
            The error obtained by derivation of the next layers.
        
        Returns:
        --------
        np.ndarray
            The error obtained by the derivation of the layer.
        """
        dinput = np.zeros(self.inputs.shape)
        error_size = error.shape[:-1]
        channels = error.shape[-1]

        for channel in range(channels):
            for raxis in range(error_size[0]):
                for caxis in range(error_size[1]):
                    dinput = self._unpool(dinput, error,
                                           (raxis, caxis, channel))

        return dinput

class MaxPoolingLayer(PoolingLayer):
    """An Layer that Pools the input by a Max factor
    
    Attributes:
    -----------
        pooling_size (int): The range of the pooling in the input.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.

        mask (np.ndarray): The coords of the selected outputs
    """
    def __init__(self, pooling_size: int):
        super().__init__(pooling_size)
        self.mask = None

    def _pool(self, window, coords):
        max_val = np.max(window)
        self.output[coords] = max_val

        pos = np.unravel_index(np.argmax(window),
                            window.shape)
        self.mask[coords] = pos

    def _unpool(self, dinput, error, coords):
        mask = tuple(self.mask[coords])
        dinput[mask] += error[coords]

        return dinput

    def feedforward(self, inputs):
        input_size = inputs.shape[:-1]
        channels = inputs.shape[-1]
        reduction = input_size - self.pooling_size + 1

        self.mask = np.empty((reduction[0], reduction[1], channels))

        super().feedforward(inputs)

        self.mask = np.array(self.mask)

        return self.output

class AvrgPoolingLayer(PoolingLayer):
    """An Layer that Pools the input by an Average factor
    
    Attributes:
    -----------
        pooling_size (int): The range of the pooling in the input.

        inputs (np.ndarray): The input values for the layer during
        feedforward.
    
        output (np.ndarray): The output of the layer after applying
        each neurons' transformations.
    """

    def _pool(self, window, coords):
        avg_val = np.average(window)
        self.output[coords] = avg_val

    def _unpool(self, dinput, error, coords):
        dinput[coords[0] : coords[0] + self.pooling_size,
               coords[1] : coords[1] + self.pooling_size,
               coords[2]] += error[coords]

        return dinput
