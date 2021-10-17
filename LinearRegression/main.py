from os import makedirs
import numpy as np
import gradient

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/concrete/"

cc_train_x = []
cc_train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_train_x.append(terms_flt[:-1])
        cc_train_y.append(terms_flt[-1])

cc_test_x = []
cc_test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_test_x.append(terms_flt[:-1])
        cc_test_y.append(terms_flt[-1])

bgd = gradient.BatchGradientDescent(cc_train_x, cc_train_y, lr = 1e-3, epochs = 250)
