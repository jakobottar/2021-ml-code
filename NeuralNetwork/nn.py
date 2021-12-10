# I got some inspiration from https://www.pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/ but the code is my own. 
import numpy as np

class sigmoid:
    def __call__(self, x: float) -> float:
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def deriv(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig * (1 - sig)

class identity:
    def __call__(self, x: float) -> float:
        return x

    def deriv(x):
        return 1

class FCLayer:
    def __init__(self, in_channels, out_channels, activation_function, include_bias = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation_function == 'sigmoid': self.activation_function = sigmoid()
        elif activation_function == 'identity': self.activation_function = identity()
        else: print("something broke")
        
        if include_bias:
            self.layer_weights = np.zeros((self.in_channels+1, self.out_channels+1))
        else:
            self.layer_weights = np.zeros((self.in_channels+1, self.out_channels))
    
    def eval(self, x):
        return self.activation_function(np.dot(x, self.layer_weights))
    
    def deriv(self, prev, zs):
        next = []
        for i in range(len(self.nodelist)):
            to_parent = self.nodelist[i].deriv(prev[i+1], zs[i])
            next.append(to_parent)

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x): 
        A = [np.atleast_2d(x)]

        for l in range(len(self.layers)):
            out = self.layers[l].eval(A[l])
            A.append(out)

        return A[-1], A

    def backward(self, A, y, alpha = 0.01):
        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value
        error = A[-1] - y

        # from here, we need to apply the chain rule and build our
        # list of deltas 'D'; the first entry in the deltas is
        # simply the error of the output layer times the derivative
        # of our activation function for the output value
        activ = sigmoid()
        D = [error * activ.deriv(A[-1])]

        # once you understand the chain rule it becomes super easy
        # to implement with a 'for' loop -- simply loop over the
        # layers in reverse order (ignoring the last two since we
        # already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * activ.deriv(A[layer])
            D.append(delta)
    
          # since we looped over our layers in reverse order we need to
        # reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] += -alpha * A[layer].T.dot(D[layer])