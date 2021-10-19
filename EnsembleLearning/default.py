from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
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
    random.shuffle(idx)

    default_train = default[idx[:24000]]
    default_test = default[idx[24000:]]

    print("training a tree...")
    tree = DecisionTree()
    tree.makeTree(default_train)

    pred_train = []
    for d in default_train:
        pred_train.append(tree.predict(d))
    train_tree = error(pred_train, [d['label'] for d in default_train])

    pred_test = []
    for d in default_test:
        pred_test.append(tree.predict(d))
    test_tree = error(pred_test, [d['label'] for d in default_test])

    print(f"tree: training error = {train_tree}, testing error = {test_tree}")

    print("running bagged trees...")
    x_pts = list(range(1,25)) + list(range(25,100,5)) + list(range(100, 550, 50))
    # x_pts = [1, 100, 500]
    train_bag = []
    test_bag = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = ensemble.BaggedTrees()
        bag.train(default_train, num_trees=x, num_samples=1000)

        train_pred = bag.predict(default_train)
        train_bag.append(error(train_pred, [d['label'] for d in default_train]))

        test_pred = bag.predict(default_test)
        test_bag.append(error(test_pred, [d['label'] for d in default_test]))
    
    print("running random forests..")
    train_rf= []
    test_rf = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = ensemble.BaggedTrees()
        bag.train(default_train, num_trees=x, num_samples=1000)

        train_pred = bag.predict(default_train)
        train_rf.append(error(train_pred, [d['label'] for d in default_train]))

        test_pred = bag.predict(default_test)
        test_rf.append(error(test_pred, [d['label'] for d in default_test]))

    print(f"bagged trees: training error = {train_bag[-1]}, testing error = {test_bag[-1]}")
    print(f"random forest: training error = {train_rf[-1]}, testing error = {test_rf[-1]}")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_pts, train_bag, color = 'tab:green', label = "training, bagged trees")
    ax.plot(x_pts, test_bag, color = 'tab:blue', label = "testing, bagged trees")
    ax.plot(x_pts, train_rf, color = 'tab:red', label = "training, random forest")
    ax.plot(x_pts, test_rf, color = 'tab:orange', label = "testing, random forest")
    ax.legend()
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")

    plt.savefig("./out/default_error.png")