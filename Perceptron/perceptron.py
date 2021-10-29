import numpy as np

class Perceptron:
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.weights = np.ndarray
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            for xi, yi in zip(X, y):
                if yi * np.dot(self.weights, xi) <= 0:
                    self.weights += r*(yi*xi)
    
    def predict(self, X) -> np.ndarray:
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class VotedPerceptron(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.votes = list
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        m = 0
        weights = [np.zeros_like(X[0])]
        cs = [0]

        for e in range(epochs):
            for xi, yi in zip(X, y):
                if yi * np.dot(weights[m], xi) <= 0:
                    weights[m] += r*(yi*xi)
                    weights.append(weights[m].copy())
                    m += 1
                    cs.append(1)
                else: cs[m] += 1

        self.votes = list(zip(weights, cs))
    
    def predict(self, X) -> np.ndarray:
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(preds)):
            inner = 0
            for w, c in self.votes:
                inner += c * np.sign(np.dot(w, X[i]))
            preds[i] = np.sign(inner)
        return preds

class AveragedPerceptron(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.a = np.ndarray
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        self.a = np.zeros_like(X[0])
        weights = np.zeros_like(X[0])

        for e in range(epochs):
            for xi, yi in zip(X, y):
                if yi * np.dot(weights, xi) <= 0:
                    weights += r*(yi*xi)
                self.a = self.a + weights

    def predict(self, X) -> np.ndarray:
        pred = lambda d : np.sign(np.dot(self.a, d))
        return np.array([pred(xi) for xi in X])