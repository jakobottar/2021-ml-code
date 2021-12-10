import numpy as np

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def identity(x):
    return x

class Node:
    def __init__(self, weights, activation_function) -> None:
        self.activation_function = activation_function
        self.weights = weights

    def eval(self, x):
        return self.activation_function(np.dot(self.weights, x))

class FCLayer:
    def __init__(self, in_channels, out_channels, activation_function):
        self.nodelist = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation_function == 'sigmoid': self.activation_function = sigmoid 
        elif activation_function == 'identity': self.activation_function = identity
        else: print("something broke")
        self.layer = self._make_layer()

    def _make_layer(self):
        for i in range(self.out_channels):
            node = Node(np.zeros((self.in_channels+1), dtype=float), self.activation_function)
            self.nodelist.append(node)
    
    def eval(self, x):
        res = [1]
        for n in self.nodelist:
            res.append(n.eval(x))
        return res


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x): 
        x = np.append(1, x)
        for l in self.layers:
            print(x)
            x = l.eval(x)
        return x[1:]

    def backward(self, pred, target):
        # we'll just use square loss here, no choice of loss function for now
        dLdy = pred - target