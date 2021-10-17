import numpy as np

class GradientDescent:
    def __init__(self, weights: list):
        self.weights = weights

    def __str__(self) -> str:
        return str(self.weights)

    def predict(self, x):
        print("yo")

# def MSE(pred, target):

    
def BatchGradientDescent(x, y, lr: float = 1, epochs: int = 10): # out = GradientDescent
    ## TODO:

    # initialize weights
    w = np.ones_like(x[0])

    # for T epochs...
    for ep in range(epochs):
        # compute gradient of J(w) at w^t
        delJ = np.zeros_like(w)

        for j in range(len(delJ)):
            for xi, yi in zip(x, y):
                delJ[j] -= (yi - np.dot(w,xi)) * xi[j]

        # update weights
        w = w - lr * delJ

        # compute loss
        loss = 0
        for xi, yi in zip(x, y):
            loss += (yi - np.dot(w, xi))**2
        loss /= 2

        print(f"epoch: {ep}, loss: {loss}")

    return GradientDescent(w)

