from os import makedirs
# import csv
import numpy as np
import svm

np.random.seed(33)

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/bank-note/"

train_x = []
train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(-1 if terms_flt[-1] == 0 else 1)

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(-1 if terms_flt[-1] == 0 else 1)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("==== Primal SVM, a) ====")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    lnot, a = 1, 1
    lr_schedule = lambda e : lnot / (1 + (lnot/a)*e)
    psvm = svm.PrimalSVM(train_x, train_y, lr_schedule=lr_schedule, C=C, epochs=100)
    print(f"learned weights: {psvm.weights}")
    print(f"training accuracy: {np.mean(train_y == psvm.predict(train_x))}")
    print(f"testing accuracy: {np.mean(test_y == psvm.predict(test_x))}")

print("==== Dual SVM ====")
#TODO:

