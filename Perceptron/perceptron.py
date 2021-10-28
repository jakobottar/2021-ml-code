import numpy as np

class PerceptronObject:
    def __init__(self, weights: np.ndarray):
        self.weights = weights
    
    def __str__(self) -> str:
        return str(self.weights)

    def predict(self, X) -> np.ndarray:
        #TODO: make this simpler
        return np.array(list(map(lambda d : np.sign(np.dot(self.weights, d)), X)))

def Perceptron(X, y, r:float = 1e-3, epochs: int=10) -> PerceptronObject:
    w = np.zeros_like(X[0])

    for e in range(epochs):
        for xi, yi in zip(X, y):
            yprime = np.sign(np.dot(w, xi))
            if yi != yprime:
                w = w + r*(yi*xi)
    
    return PerceptronObject(w)