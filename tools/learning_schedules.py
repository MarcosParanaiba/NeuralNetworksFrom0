"""The basic Learning Schedules"""
from abc import ABC, abstractmethod
import numpy as np

class LearningSchedule(ABC):
    """An Abstract Schedule"""
    @abstractmethod
    def __init__(self, initial_rate: np.ndarray):
        self.initial_rate = initial_rate

    @abstractmethod
    def setlearning(self, epoch: int):
        """Get a new learning rate from the epoch."""

class NoSchedule(LearningSchedule):
    """An Fixed Schedule"""
    def __init__(self, initial_rate: np.ndarray):
        self.initial_rate = initial_rate

    def setlearning(self, _):
        return self.initial_rate
