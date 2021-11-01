# Perceptron

## `main.py`
This is the main driver code, it reads in the given dataset and converts it to a numpy array of floats.
To follow the convention of `sklearn` functions, I broke off the labels `y` from the features `X` and pass them into the function independently. 
This allows for greater future customization.

`main.py` then runs all three perceptron variants, reporting learned weight vectors and training/testing accuracy. 
My implementation returns a `numpy.ndarray`, making the average accuracy easy to compute by checking equality with the true `y` array and taking an average. 
For the Voted Perceptron, the list of weights and counts is fairly large, so I save it as a csv for submission. This file is created as `./out/vp_weights.csv`.

## Standard Perceptron

## Voted Perceptron

## Averaged Perceptron