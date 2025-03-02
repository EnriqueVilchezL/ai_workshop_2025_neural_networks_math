from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    @abstractmethod
    def compute(self, tensors : dict[str, np.ndarray]) -> float:
        ...

class Accuracy(Metric):
    def compute(self, tensors : dict[str, np.ndarray]) -> float:
        y = tensors['Y']
        y_hat = tensors['Y_hat']

        return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))