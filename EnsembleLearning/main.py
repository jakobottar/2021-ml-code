from os import makedirs
import ensemble
import numpy as np

import matplotlib.pyplot as plt

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/bank/"

def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)

def HandleLine(line):
    terms = line.strip().split(",")
    t_dict = { # TODO: better way of doing this?
        "age": int(terms[0]), # numeric
        "job": terms[1], # categorical
        "marital": terms[2], # categorical
        "education": terms[3], # categorical
        "default": terms[4], # binary
        "balance": int(terms[5]), #numeric
        "housing": terms[6], # binary
        "loan": terms[7], # binary
        "contact": terms[8], # categorical
        "day": int(terms[9]), # numeric
        "month": terms[10], # categorical
        "duration": int(terms[11]), # numeric
        "campaign": int(terms[12]), # numeric
        "pdays": int(terms[13]), # numeric
        "previous": int(terms[14]), # numeric 
        "poutcome": terms[15], # categorical

        "label": terms[16] # binary
    }
    return t_dict

if __name__ == '__main__':
    train_bank = []
    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            train_bank.append(HandleLine(line))

    test_bank = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            test_bank.append(HandleLine(line))

    print("datasets loaded")

    # ada = ensemble.AdaBoost()
    # ada.train(train_dataset, 4)

    # train_pred = ada.predict(train_dataset)
    # print(error(train_pred, [d['label'] for d in train_dataset]))

    # test_pred = ada.predict(test_dataset)
    # print(error(test_pred, [d['label'] for d in test_dataset]))

    print("running bagged trees...")
    x_pts = [1,2,3,4,5,7,10,12,14,16,32,64, 128, 256, 512]
    train_err = []
    test_err = []

    for x in x_pts:
        print(x)

        bag = ensemble.BaggedTrees()
        bag.train(train_bank, num_trees=x, num_samples=1000)

        train_pred = bag.predict(train_bank)
        train_err.append(error(train_pred, [d['label'] for d in train_bank]))

        test_pred = bag.predict(test_bank)
        test_err.append(error(test_pred, [d['label'] for d in test_bank]))

    plt.plot(x_pts, train_err, x_pts, test_err)
    plt.savefig("./out/bagged_error.png")