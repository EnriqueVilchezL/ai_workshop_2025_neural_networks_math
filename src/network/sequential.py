import numpy as np
from network.network_module import NetworkModule
import pickle as pkl

class Sequential(NetworkModule):
    def __init__(self, layers : list[NetworkModule]) -> None:
        super().__init__()
        self.submodules = layers

    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        x = self.tensors['X']

        for module in self.submodules:
            module({'X': x})
            x = module.tensors['Y']
        
        self.tensors['Y'] = x
        return self.tensors['Y']

    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        dy = outputs_gradients['dY']

        for module in reversed(self.submodules):
            module.backward({'dY': dy})
            dy = module.gradients['dX']

        self.gradients['dX'] = dy

    def save(self, path : str) -> None:
        with open(path, 'wb') as file:
            pkl.dump(self, file)
    
    @staticmethod
    def load(path : str) -> "Sequential":
        with open(path, 'rb') as file:
            model : Sequential = pkl.load(file)
        return model
