from os import makedirs
from posix import times_result
from typing import Dict
import numpy as np
import random

from DecisionTree import DecisionTree
import ensemble

# from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dataset_loc = "../data/default/"

def array2Dict(data, header):
    out = [None] * len(data)

    for i, d in enumerate(data):
        out[i] = {}
        for j, label in enumerate(header):
            try: val = int(d[j])
            except ValueError: val = d[j]
            out[i][label] = val

    return out

def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)

if __name__ == '__main__':
    default_raw = []
    with open(dataset_loc + "default.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            default_raw.append(terms)

    default = np.array(array2Dict(default_raw[1:], default_raw[0]))
    idx = list(range(len(default)))
    # random.shuffle(idx)

    default_train = default[idx[:24000]]
    default_test = default[idx[24000:]]

    print("training a tree...")
    tree = DecisionTree()
    tree.makeTree(default_train)

    pred_train = []
    for d in default_train:
        pred_train.append(tree.predict(d))
    error_train = error(pred_train, [d['label'] for d in default_train])

    pred_test = []
    for d in default_test:
        pred_test.append(tree.predict(d))
    error_test = error(pred_test, [d['label'] for d in default_test])

    print(f"training error: {error_train}, testing error: {error_test}")

    # print("training adaboost...")
    ## TODO:

    print("training bagged trees...")
    bag = ensemble.BaggedTrees()
    bag.train(default_train, num_trees=500, num_samples=5000)

    pred_train = bag.predict(default_train)
    error_train = error(pred_train, [d['label'] for d in default_train])

    pred_test = bag.predict(default_test)
    error_test = error(pred_test, [d['label'] for d in default_test])

    print(f"training error: {error_train}, testing error: {error_test}")

    print("training random forest...")
    rf = ensemble.RandomForest()
    rf.train(default_train, num_trees=500, num_samples=5000)

    pred_train = rf.predict(default_train)
    error_train = error(pred_train, [d['label'] for d in default_train])

    pred_test = rf.predict(default_test)
    error_test = error(pred_test, [d['label'] for d in default_test])

    print(f"training error: {error_train}, testing error: {error_test}")