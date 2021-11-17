import numpy as np
import math
from numpy.lib.function_base import disp
import scipy.optimize

# def _batch_dataloader(X, y, batch_size, shuffle = True):
#     idxs = np.arange(len(X))
#     if shuffle:
#         np.random.shuffle(idxs)
#     batched_data = []

#     for i in range(math.ceil(len(X)/batch_size)):
#         lower = batch_size * i
#         upper = min(len(X), batch_size * (i+1))
#         batch = {'y': y[idxs[lower:upper]], 
#                  'X': X[idxs[lower:upper], :]}
#         batched_data.append(batch)
    
#     return batched_data

class PrimalSVM:
    def __init__(self, X, y, lr_schedule, C, epochs=10):
        self.weights = np.ndarray
        self.train(X, y, lr_schedule, C, epochs)

    def train(self, X, y, lr_schedule, C, epochs=10):
        X = np.insert(X, 0, [1]*len(X), axis=1)
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            lr = lr_schedule(e)
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)

            for i in idxs:
                if y[i]*np.dot(self.weights, X[i]) <= 1:
                    self.weights = self.weights - lr*self.weights + lr*C*y[i]*X[i]
                else:
                    self.weights = (1-lr)*self.weights

    def predict(self, X) -> np.ndarray:
        X = np.insert(X, 0, [1]*len(X), axis=1)
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class DualSVM:
    def __init__(self, X, y, C, kernel = "dot"):
        self.wstar = np.ndarray
        self.bstar = float
        self.train(X, y, C, kernel)

    def train(self, X, y, C, kernel = "dot"):

        if kernel == "dot":
            kernel_func = np.dot
        else: 
            NotImplementedError

        def minim(a, X, y):
            val = 0
            for i in range(len(X)):
                for j in range(len(X)):
                    val += y[i]*y[j]*a[i]*a[j]*kernel_func(X[i], X[j])
                    
            return 0.5*val - np.sum(a)

        constraints = [
            {
                'type': 'ineq',
                'fun': lambda a : a
            },
            {
                'type': 'ineq',
                'fun': lambda a: C - a
            },
            {
                'type': 'eq',
                'fun': lambda a: np.sum(a*y)
            },
        ]

        res = scipy.optimize.minimize(minim, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP', constraints=constraints)
        self.wstar = np.zeros_like(X[0])
        for i in range(len(X)):
            self.wstar += res['x'][i]*y[i]*X[i]

        self.bstar = 0
        for j in range(len(X)):
            self.bstar += y[j] - np.dot(self.wstar, X[j])
        self.bstar /= len(X)

    def predict(self, X) -> np.ndarray:
        pred = lambda d : np.sign(np.dot(self.wstar, d) + self.bstar)
        return np.array([pred(xi) for xi in X])

