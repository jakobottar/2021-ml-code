# Perceptron

## `main.py`
This is the main driver code, it reads in the given dataset and converts it to a numpy array of floats.
To follow the convention of `sklearn` functions, I broke off the labels `y` from the features `X` and pass them into the function independently. 
This allows for greater future customization.

`main.py` then runs all three perceptron variants, reporting learned weight vectors and training/testing accuracy. 
My implementation returns a `numpy.ndarray`, making the average accuracy easy to compute by checking equality with the true `y` array and taking an average. 
For the Voted Perceptron, the list of weights and counts is fairly large, so I save it as a csv for submission. This file is created as `./out/vp_weights.csv`.

## Standard Perceptron
I implemeted the Standard Perceptron algorithm in `Perceptron/perceptron.py` as `Perceptron()`. 
The object can be created then trained, or trained upon creation. 
The training function, `Perceptron.train()` first appends a bias term to the numpy array of features, then trains for `num_epochs` epochs, updating the weight at each element of the training dataset using the learning rate `r`.  

Prediction on the trained model follows the algorithm, returning a `numpy.ndarray` of the same length as the input dataset `X`. 
## Voted Perceptron
The Voted Perceptron inherets `Perceptron`, and changes the initialization function for the `votes` array, the training function to follow the Voted Perceptron algorithm, as well as the prediction function similarly. 

## Averaged Perceptron
Again, the Averaged Perceptron inherets `Perceptron`, but since it is much closer to the original, it only changes the training function. 