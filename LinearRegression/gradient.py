import numpy as np

def MSE(pred, target) -> float:
    assert len(pred) == len(target)
    pred, target = np.array(pred), np.array(target)
    return np.sum((target - pred)**2) / 2

class LMSWeights:
    def __init__(self, weights: list):
        self.weights = weights

    def __str__(self) -> str:
        return str(self.weights)

    def predict(self, x) -> list:
        return list(map(lambda d : np.dot(self.weights, d), x))
    
def BatchGradientDescent(x, y, lr: float = 1, epochs: int = 10, threshold = 1e-6):

    # initialize weights
    w = np.ones_like(x[0])

    losses, lastloss, diff = [], 9999, 1
    # for T epochs...
    for ep in range(epochs):
        if diff <= threshold: break
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
        
        diff = abs(loss - lastloss)
        lastloss = loss
        losses.append(loss)

    print(f"converged at epoch {ep} to {diff}")
    return LMSWeights(w), losses

def StochasticGradientDescent(x, y, lr: float = 1, epochs: int = 10, threshold = 1e-6):

    # batched = []
    # idx = 0
    # data = list(zip(x, y))
    # while idx <= len(data):
    #     batch = data[idx:min(idx+batch_size, len(data))]
    #     batched.append(batch)
    #     idx += batch_size

    w = np.ones_like(x[0])

    losses, lastloss, diff = [], 9999, 1
    for ep in range(epochs):
        if diff <= threshold: break
        # for each element, update weights
        for xi, yi in zip(x, y):
            for j in range(len(w)):
                w[j] += lr * (yi - np.dot(w, xi)) * xi[j]

            # compute loss
            loss = 0
            for xi, yi in zip(x, y):
                loss += (yi - np.dot(w, xi))**2
            loss /= 2
            
            diff = abs(loss - lastloss)
            lastloss = loss
            losses.append(loss)

    print(f"converged at epoch {ep} to {diff}")
    return LMSWeights(w), losses

def LMSRegression(x, y):
    x = np.transpose(np.array(x))
    y = np.array(y)

    w = np.linalg.inv(x @ np.transpose(x)) @ (x @ y)
    return LMSWeights(w)