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

class TimeBasedSchedule(LearningSchedule):
    """An Time Based Schedule"""
    def __init__(self, initial_rate, decay):
        self.initial_rate = initial_rate
        self.decay = decay

    def setlearning(self, epoch):
        return self.initial_rate * ((1 + (self.decay * epoch))
                                    ** (-epoch - 1) )

class StepBasedSchedule(LearningSchedule):
    """An Step Based Schedule"""
    def __init__(self, initial_rate, decay, step):
        self.initial_rate = initial_rate
        self.decay = decay
        self.step = step

    def setlearning(self, epoch):
        return self.initial_rate * (self.decay **
                                    np.floor((1 + epoch) / self.step))
