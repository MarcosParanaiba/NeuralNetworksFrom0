"""The basic functions to use in the Neurons"""
from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):
    """Base for functions"""
    @classmethod
    @abstractmethod
    def f(cls, x):
        """The call of the function"""

    @classmethod
    @abstractmethod
    def d(cls, x):
        """The derivation of the function"""

class Sigmoid(Function):
    """A sigmoid function"""
    @classmethod
    def f(cls, x):
        x = np.clip(x, -500, 500)
        return 1/(1 + np.exp(-x))

    @classmethod
    def d(cls, x):
        return cls.f(x) * (1 - cls.f(x))

class TanH(Function):
    """A TanH function"""
    @classmethod
    def f(cls, x):
        return np.tanh(x)

    @classmethod
    def d(cls, x):
        return 1 - (np.tanh(x)**2)

class reLU(Function):
    """A reLU"""
    @classmethod
    def f(cls, x):
        return max(0, x)

    @classmethod
    def d(cls, x):
        return max(0, x) / x
