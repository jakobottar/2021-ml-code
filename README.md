# 2021-ml-code
This is a machine learning library developed by Jakob Johnson for CS 6350 at The University of Utah

## Decision Tree - Homework 1

<!-- TODO: -->

## Ensemble Learning - Homework 2

<!-- TODO: -->

## Linear Regression - Homework 2

<!-- TODO:  -->

## Perceptron - Homework 3

### `main.py`
This is the main driver code, it reads in the given dataset and converts it to a numpy array of floats.
To follow the convention of `sklearn` functions, I broke off the labels `y` from the features `X` and pass them into the function independently. 
This allows for greater future customization.

`main.py` then runs all three perceptron variants, reporting learned weight vectors and training/testing accuracy. 
My implementation returns a `numpy.ndarray`, making the average accuracy easy to compute by checking equality with the true `y` array and taking an average. 
For the Voted Perceptron, the list of weights and counts is fairly large, so I save it as a csv for submission. This file is created as `./out/vp_weights.csv`.

### Standard Perceptron
I implemeted the Standard Perceptron algorithm in `Perceptron/perceptron.py` as `Perceptron()`. 
The object can be created then trained, or trained upon creation. 
The training function, `Perceptron.train()` first appends a bias term to the numpy array of features, then trains for `num_epochs` epochs, updating the weight at each element of the training dataset using the learning rate `r`.  

Prediction on the trained model follows the algorithm, returning a `numpy.ndarray` of the same length as the input dataset `X`. 
### Voted Perceptron
The Voted Perceptron inherets `Perceptron`, and changes the initialization function for the `votes` array, the training function to follow the Voted Perceptron algorithm, as well as the prediction function similarly. 

### Averaged Perceptron
Again, the Averaged Perceptron inherets `Perceptron`, but since it is much closer to the original, it only changes the training function. 

## SVM - Homework 4

### `main.py`
This is the main driver code, in it I load the training and testing data, formatting them as numpy arrays of floats. Similarly to the last assignment, I broke off the labels `y` from the features `X` and pass them into the function independently, to follow the convention of `sklearn` functions.
I then train the models using the hyperparameters from the assignment page. I manually tested different learning rate and gamma values, settling on 1 and 1 as the best values. Finally, I compute the training and testing predictions and report their accuracies. 
For the parts of the problems that require them, I report the number of saved support vectors, as well as compare overlap. 

### Primal SVM
I implented the Primal SVM algorithm as `PrimalSVM` in `svm.py`. It follows the primal optimization algorithm, using stochastic gradient descent to find optimal weights. To help with speedup between epochs, instead of manually shuffling the dataset, which makes maintaining a link to the y-s hard, I created a vector of indicies to read from, then shuffled that, allowing the dataset to remain untouched while we read in a random order from it. 
Following the last assignment, I wanted to return an object holding the weights with a `predict()` method, allowing us to easily test it on a dataset. 

### Dual SVM
I implented the Dual SVM algorithm as `DualSVM` in `svm.py`. It again follows the algorithm presented in class, utilizing `scipy.optimize.minimize` to perform the quadratic optimization. This implementation supports the use of the identity (or dot product) kernel as well as a Gaussian kernel.
Using the appropriate kernel, it generates the objective function to minimize as well as the dict of constraints the optimizer requires. 
The inner optimization needed to be adapted to `numpy` functions to speed up the optimization algorithm. Multiplying the a and y vectors by a square matrix of ones results in the proper pairwise multiplication operation. 

For the Gaussian kernel, I had to expand the kernel formula, expanding `(x-y)^2` to `x^2 - 2xy + y^2` and adding the gamma and exponential terms after. 

Finally, multiplying these together and taking the sum results in the proper value, significantly faster than nested for loops.

```
ymat = y * np.ones((len(y), len(y)))
amat = a * np.ones((len(a), len(a)))

if kernel == 'dot':
    xvals = (X@X.T)
if kernel == 'gaussian':
    xvals = X**2@np.ones_like(X.T) - 2*X@X.T + np.ones_like(X)@X.T**2 
    xvals = np.exp(-( xvals / gamma))

vals = (ymat*ymat.T) * (amat*amat.T) * xvals
return 0.5*np.sum(vals) - np.sum(a)
```

After optimization, we use the formulas derived to calculate `wstar` (the weights) and `bstar` (the bias term). Finally, we determine all support vectors by their Lagrange values, throwing away any within 1E-10 from 0, assuming those to be floating point errors. 

## Neural Networks - Homework 5

### Driver Code
The driver code for the homework questions is located in `testing_net.py`, `random_init.py`, and `zeroes_init.py`. These represent the code for question 2 parts a, b, and c respectively. They are fairly straightforward, loading the data and creating the networks of various sizes with the proper initialization. They train the networks and report plots of the training error. 

### Neural Network
The neural network code is located in `nn.py`. I wanted to copy the `pytorch` method of creating a network using layers so you can easily control what's inside the network. The `NeuralNetwork` class takes in an array of `FCLayer` objects which form the architecture of the network. 

Most of the work goes on inside the `FCLayer` objects. They can be of any size, but must flow from one to another shape-wise. They hold a matrix of weights whose initialization can be either random from a standard normal distribution or all zeros. They also have an activation function that we define with a derivative. The final output layer does not need a bias term, so we also have an option to turn that off. These `FCLayer` objects also have methods to facilitate a forward pass through the network, as well as backward pass partial derivative calculation and weight updates through gradient calculation using matrix multiplication with the weight matrix. 

During a forward pass of the network, we simply pass the values through each layer of the network, recording the Zs (activation values) at each layer for backpropagation. 

For the backwards pass, we first calculate the partial of the loss function, which in this case is `y-ystar` because we're only using squared loss. We then go through each layer in reverse order and calculate a matrix of partial derivatives for each layer using our saved activation values and the derivative of the activation function. Then finally we go back through and update the weights, calculating the gradients by multiplying the partial derivative matrix with the activation values. 

The backpropagation algorithm is not that complex, however I had a tough time figuring out what data structure to use to hold the values, weights, and derivatives. I came across a method that mentioned using matrices for the weights and partials, which helped significantly. This is probably much much faster than a graph-traversal-based method and lends itself to GPU accelleration, just like `pytorch`. 
