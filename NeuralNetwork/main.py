import nn
import numpy as np

net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 3, out_channels = 3, activation_function = 'sigmoid'),
    nn.FCLayer(in_channels = 3, out_channels = 3, activation_function = 'sigmoid'),
    nn.FCLayer(in_channels = 3, out_channels = 1, activation_function = 'identity')
])

y = net.forward(np.array([1,1,1]))
print(y)