import pickle
import numpy as np
from typing import Tuple

def load(path : str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path,'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
