import numpy as np

class Perceptron:
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.weights = np.ndarray
        self.train(X, y, r, epochs)

    def __str__(self) -> str:
            return str(self.weights)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            for xi, yi in zip(X, y):
                yprime = np.sign(np.dot(self.weights, xi))
                if yi != yprime:
                    self.weights += r*(yi*xi)
    
    def predict(self, X) -> np.ndarray:
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class VotedPerceptron(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.weights = list
        self.cs = list
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        m = 0
        self.weights = [np.zeros_like(X[0])]
        self.cs = [0]

        for e in range(epochs):
            for xi, yi in zip(X, y):
                if yi * np.dot(self.weights[m], xi) <= 0:
                    self.weights[m] += r*(yi*xi)
                    self.weights.append(self.weights[m].copy())
                    m += 1
                    self.cs.append(1)
                else: self.cs[m] += 1
    
    def predict(self, X) -> np.ndarray:
        preds = [0] * len(X)
        for i in range(len(preds)):
            inner = 0
            for c, w in zip(self.cs, self.weights):
                inner += c * np.sign(np.dot(w, X[i]))
            preds[i] = 1 if inner >= 0 else -1
        return np.array(preds)