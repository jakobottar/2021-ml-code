import nn
import numpy as np

net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 2, out_channels = 2, activation_function = 'sigmoid'), # input
    nn.FCLayer(in_channels = 2, out_channels = 2, activation_function = 'sigmoid'), # hidden
    nn.FCLayer(in_channels = 2, out_channels = 1, activation_function = 'identity', include_bias=False) # output
])

y, A = net.forward(np.array([1, 1,1]))
print(y, A)
# net.backward(y, 1)
# print(net.W)