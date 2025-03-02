import numpy as np
from network.network_module import NetworkModule

class Activation(NetworkModule):
    pass
    
class Sigmoid(Activation):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        self.tensors['Y'] = 1 / (1 + np.exp(-self.tensors['X']))
        return self.tensors['Y']
    
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        self.gradients['dX'] = outputs_gradients['dY'] * self.tensors['Y'] * (1 - self.tensors['Y'])

class ReLU(Activation):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        self.tensors['Y'] = np.maximum(0, self.tensors['X'])
        return self.tensors['Y']
    
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        self.gradients['dX'] = outputs_gradients['dY'] * (self.tensors['X'] > 0)

class Tanh(Activation):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        self.tensors['Y'] = np.tanh(self.tensors['X'])
        return self.tensors['Y']
    
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        self.gradients['dX'] = outputs_gradients['dY'] * (1 - self.tensors['Y'] ** 2)

class Softmax(Activation):
    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        exps = np.exp(self.tensors['X'] - np.max(self.tensors['X'], axis=1, keepdims=True))
        self.tensors['Y'] = exps / np.sum(exps, axis=1, keepdims=True)
        return self.tensors['Y']
    
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        self.gradients['dX'] = np.empty_like(outputs_gradients['dY'])
        
        for index, (y, dy) in enumerate(zip(self.tensors['Y'], outputs_gradients['dY'])):
            y = y.reshape(-1, 1)
            jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
            self.gradients['dX'][index] = np.dot(jacobian_matrix, dy)