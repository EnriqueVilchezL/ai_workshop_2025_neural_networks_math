import numpy as np
from abc import ABC, abstractmethod

class NetworkModule(ABC):
    def __init__(self) -> None:
        self.tensors : dict[str, np.ndarray] = {
            'X': None,
            'Y': None
        }
        self.gradients : dict[str, np.ndarray] = {
            'dX': None
        }
        self.submodules : list[NetworkModule] = None
        self.parameters : dict[str, np.ndarray] = None

    def __call__(self, *args, **kwds) -> np.ndarray:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        ...
    
    @abstractmethod
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        ...