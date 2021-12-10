import nn
import numpy as np

net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 2, out_channels = 2, activation_function = 'sigmoid'),
    nn.FCLayer(in_channels = 2, out_channels = 2, activation_function = 'sigmoid'),
    nn.FCLayer(in_channels = 2, out_channels = 1, activation_function = 'identity')
])

y = net.forward(np.array([1,1]))
print(y)