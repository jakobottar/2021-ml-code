import numpy as np


class sigmoid:
    def __call__(self, x: float) -> float:
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def deriv(x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig * (1 - sig)

class identity:
    def __call__(self, x: float) -> float:
        return x

    def deriv(x):
        return 1

class Node:
    def __init__(self, weights, activation_function) -> None:
        self.activation_function = activation_function
        self.weights = weights

    def eval(self, x):
        return self.activation_function(np.dot(self.weights, x))

    # zs is the set of zs that the function recieved in the forward pass
    def deriv(self, from_child, zs):
        grads = np.zeros_like(self.weights)
        to_parent = np.zeros_like(self.weights) 
        dsigmads = self.activation_function.deriv(np.dot(self.weights, zs))

        for j, c in enumerate(from_child):          
            for i, w in enumerate(self.weights):
                grads[i] += c * dsigmads * zs[i]
                to_parent[i] += c * dsigmads * w

            print(grads)
            to_parent = from_child * dsigmads * self.weights
            return to_parent

class FCLayer:
    def __init__(self, in_channels, out_channels, activation_function):
        self.nodelist = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation_function == 'sigmoid': self.activation_function = sigmoid()
        elif activation_function == 'identity': self.activation_function = identity()
        else: print("something broke")
        self.layer = self._make_layer()

    def _make_layer(self):
        for i in range(self.out_channels):
            node = Node(np.zeros((self.in_channels+1), dtype=float), self.activation_function)
            self.nodelist.append(node)
    
    def eval(self, x):
        res = np.array([1])
        for n in self.nodelist:
            res = np.append(res, n.eval(x))
        return res
    
    def deriv(self, prev, zs):
        next = []
        for i in range(len(self.nodelist)):
            to_parent = self.nodelist[i].deriv(prev[i+1], zs[i])
            next.append(to_parent)

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x): 
        zs = [x]
        x = np.append(1, x)
        for l in self.layers:
            x = l.eval(x)
            zs.append(x)
        res = x[1:]
        if len(res) == 1: return res[0], zs
        return res, zs # this is a bit messy, ideally we'd use some kind of class here

    def backward(self, pred, target):
        # we'll just use square loss here, no choice of loss function for now
        zs = pred[1]
        dLdy = np.array([pred[0] - target])
        for i in range(len(self.layers)-1, -1, -1):
            print(i)