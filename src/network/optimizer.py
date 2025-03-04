import numpy as np
from abc import ABC, abstractmethod
from network.layer import Layer
from network.network_module import NetworkModule

class Optimizer(ABC):
    def __init__(self, learning_rate : float) -> None:
        self.learning_rate = learning_rate
    
    def update(self, module : NetworkModule) -> None:
        stack = [module]

        while stack:
            current_module = stack.pop()
            for submodule in current_module.submodules:
                if submodule.submodules is not None:
                    stack.append(submodule)
                
                if submodule.parameters is not None:
                    self.update_module(submodule)

    @abstractmethod
    def update_module(self, module : NetworkModule) -> None:
        ...

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate : float) -> None:
        super().__init__(learning_rate)
    
    def update_module(self, module : NetworkModule) -> None:
        for parameter in module.parameters.keys():
            module.parameters[parameter] -= self.learning_rate * module.gradients[f'd{parameter}']