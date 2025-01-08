import numpy as np
from network.network_module import NetworkModule

class Loss(NetworkModule):
    pass

class MeanSquaredError(Loss):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['Y'] = inputs['Y']
        self.tensors['Y_hat'] = inputs['Y_hat']
        self.tensors['L'] = np.mean((self.tensors['Y'] - self.tensors['Y_hat']) ** 2, axis=1)
        return self.tensors['L']
    
    def backward(self) -> None:
        self.gradients['dY_hat'] = 2 * (self.tensors['Y_hat'] - self.tensors['Y']) / self.tensors['Y'].shape[0]

class CategoricalCrossEntropy(Loss):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['Y'] = inputs['Y']
        self.tensors['Y_hat'] = np.clip(inputs['Y_hat'], 1e-7, 1 - 1e-7)
        self.tensors['L'] = -np.sum(self.tensors['Y'] * np.log(self.tensors['Y_hat']), axis=1)
        return self.tensors['L']
    
    def backward(self) -> None:
        self.gradients['dY_hat'] = -self.tensors['Y'] / self.tensors['Y_hat']