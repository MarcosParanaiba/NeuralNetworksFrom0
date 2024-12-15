"""The basic Neural Networks"""
from typing import Type
import numpy as np
import tools.learning_schedules as s
import structures.layers as l

class NeuralNetwork:
    """An basic Neural Network"""
    def __init__(self, *args: Type[l.DenseLayer]):
        self.layers = list(args)
    def addlayer(self, *args: Type[l.DenseLayer]):
        """Adds layers to the network.
        
        Parameters:
        -----------
        *args : Type[l.DenseLayer]
            The layers to be added.
        """
        self.layers.extend(args)
    def feedforward(self, inputs: np.ndarray):
        """Compute de feedfoward of the layer.
        
        Pass the feedforward through the layers and output a result of
        the computations.

        Parameters:
        -----------
        inputs : np.ndarray
            A 1-dimensional array that will be processed by the layer.
        
        Returns:
        --------
        np.ndarray
            The output value of the transformations.
        """
        inputs = [inputs] if inputs.ndim == 1 else inputs

        for layer in self.layers:
            inputs = np.array([layer.feedforward(inp)
                               for inp in inputs])

        return inputs

    def backwardpass(self, error: np.ndarray, learning_rate: float):
        """Compute the backwardpass of the layer.

        Backpropagate the error through the layers, correcting it's
        atrributes.

        Parameters:
        -----------
        error : np.ndarray
            The error obtained by derivation of the next layers.
        learning_rate : float
            The impact of givin error in the matter of the learning.
        """
        for layer in reversed(self.layers):
            error = layer.backwardpass(error, learning_rate)
    def train(self, inputs: np.ndarray, answers: np.ndarray,
              epochs: int, learning_schedules: Type[s.LearningSchedule],
              method: str = 'stochastic'):
        """Train the network
        
        By applying the gradient descent repeatedly, decreasing the
        error, get answers more closelly to the 'perfect' answers.

        Parameters:
        -----------
        inputs : np.ndarray
            A 1-dimensional array that will be processed by the layer.
        answers : np.ndarray
            A 1-dimensional array that will be the 'perfect' answer.
        epochs : int
            How much will be the gradient descent applied.
        learning_schedules : Type[s.LearningSchedule]
            The learning schedule that will be used to get each learning
            rate across the epochs.
        method : str
            The name of the method of trainment.
                stochastic: make each answer correct at a time.
                batch: make all the answer's mean correct.

        Returns:
        --------
            np.array
                The error aross the epochs.
        """
        gradient = np.empty(epochs)

        for epoch in range(epochs):
            learning_rate = learning_schedules.setlearning(epoch)
            u = self.feedforward(inputs)

            error = answers - u
            gradient[epoch] = np.absolute(np.sum(error)/answers.shape[0])
            error = [error] if error.ndim == 1 else error

            if method == 'stochastic':
                for err in error:
                    self.backwardpass(err, learning_rate)
            elif method == 'batch':
                error = np.mean(error, axis=0)
                self.backwardpass(error, learning_rate)

        return gradient

    @property
    def weights(self):
        """Get weights."""
        return [layer.weights for layer in self.layers]
    @weights.setter
    def weights(self, value):
        """Set weights."""
        for (i, v) in enumerate(value):
            self.layers[i].weights = v
    @property
    def weights_shape(self):
        """Get the weights' shape"""
        return [layer.weights.shape for layer in self.layers]
    @property
    def biases(self):
        """Get biases."""
        return [layer.bias for layer in self.layers]
    @biases.setter
    def biases(self, value):
        """Set bias."""
        for (i, v) in enumerate(value):
            self.layers[i].bias = v
    @property
    def biases_shape(self):
        """Get the biases' shape"""
        return [layer.biases.shape for layer in self.layers]
