import numpy as np
from network.network_module import NetworkModule

class Layer(NetworkModule):
    pass

class Dense(Layer):
    def __init__(self, inputs_size : int, outputs_size : int) -> None:
        super().__init__()
        self.gradients['dW'] = None
        self.gradients['dB'] = None

        self.input_size : int = inputs_size
        self.output_size : int = outputs_size

        self.parameters = {
            'W': 0.01 * np.random.randn(inputs_size, outputs_size),
            'B': 0.01 * np.random.randn(outputs_size)
        }

    def forward(self, inputs : dict[str, np.ndarray]) -> np.ndarray:
        self.tensors['X'] = inputs['X']
        self.tensors['Y'] = np.dot(self.tensors['X'], self.parameters['W']) + self.parameters['B']
        return self.tensors['Y']
    
    def backward(self, outputs_gradients : dict[str, np.ndarray]) -> None:
        self.gradients['dX'] = np.dot(outputs_gradients['dY'], self.parameters['W'].T)
        self.gradients['dW'] = np.dot(self.tensors['X'].T, outputs_gradients['dY'])
        self.gradients['dB'] = np.sum(outputs_gradients['dY'], axis=0)