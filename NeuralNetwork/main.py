import nn
import numpy as np
from os import makedirs

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/bank-note/"

def square_loss(pred, target):
    return 0.5*(pred - target)**2

print("testing network ==========")

net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 2, out_channels = 4, activation_function = 'sigmoid'), # input
    nn.FCLayer(in_channels = 4, out_channels = 4, activation_function = 'sigmoid'), # hidden
    nn.FCLayer(in_channels = 4, out_channels = 1, activation_function = 'identity', include_bias=False) # output
])

x = np.array([1,1])
for i in range(25):
    y, A = net.forward(x)
    print(f"x = [1,1], y^* = 1, y = {y}")
    print("loss:", square_loss(y, 1))
    net.backward(A, 1)

train_x = []
train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(terms_flt[-1])

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(terms_flt[-1])

test_x = np.array(test_x)
test_y = np.array(test_y)

net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 8, activation_function = 'sigmoid'), # input
    nn.FCLayer(in_channels = 8, out_channels = 8, activation_function = 'sigmoid'), # hidden
    nn.FCLayer(in_channels = 8, out_channels = 1, activation_function = 'identity', include_bias=False) # output
])

num_epochs = 10
for e in range(num_epochs):
    print(f"training epoch [{e+1} / {num_epochs}]")
    losses = []
    idxs = np.arange(len(train_x))
    np.random.shuffle(idxs)
    for i in idxs:
        y, A = net.forward(train_x[i])
        losses.append(square_loss(y, train_y[i]))
        net.backward(A, train_y[i], 0.075)
    print(f"training accuracy: {1 - np.mean(losses)}")

losses = []
for i in range(len(test_x)):
    y, _ = net.forward(test_x[i])
    losses.append(square_loss(y, test_y[i]))
print(f"testing accuracy: {1 - np.mean(losses)}")