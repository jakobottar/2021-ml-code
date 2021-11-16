import numpy as np
import math

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
        