import ensemble
import numpy as np
import random

dataset_loc = "../data/bank/"

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

def str2Num(value):
    if value == 'no': return 0
    else: return 1

if __name__ == '__main__':
    bank_raw_train = [["age", "job", "marital", "education", 
              "default", "balance", "housing", "loan", 
              "contact", "day", "month", "duration", 
              "campaign", "pdays", "previous", "poutcome", "label"]]

    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_train.append(terms)

    bank_raw_test = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_test.append(terms)

    train_bank = np.array(array2Dict(bank_raw_train[1:], bank_raw_train[0]))
    test_bank = np.array(array2Dict(bank_raw_test, bank_raw_train[0]))
    idx = list(range(len(train_bank)))
    random.shuffle(idx)

    train_bank = train_bank[idx[:1000]]

    bagged_trees = []
    single_trees = []
    for i in range(100):
        bag = ensemble.BaggedTrees()
        bag.train(train_bank, num_trees=500, num_samples=500)
        bagged_trees.append(bag)
        single_trees.append(bag.getFirstTree())

    bias_single, bias_bagged, var_single, var_bagged = [], [], [], []
    for d in test_bank:
        bagged = list(map(lambda t : str2Num(t.predict([d])[0]), bagged_trees))
        single = list(map(lambda t : str2Num(t.predict(d)), single_trees))
        lab = str2Num(d['label'])

        bias_single.append((lab - np.mean(single)) ** 2)
        bias_bagged.append((lab - np.mean(bagged)) ** 2)
        var_single.append(np.std(single) ** 2)
        var_bagged.append(np.std(bagged) ** 2)

    print(f"single tree: \n    bias: {np.mean(bias_single)}\n    variance: {np.mean(var_single)}\n    GSE: {np.mean(bias_single) + np.mean(var_single)}")
    print(f"bagged trees: \n    bias: {np.mean(bias_bagged)}\n    variance: {np.mean(var_bagged)}\n    GSE: {np.mean(bias_bagged) + np.mean(var_bagged)}")

### Results (long runtime!)
# single tree: 
#     bias: 0.10130452000000001
#     variance: 0.09764348
#     GSE: 0.198948
# bagged trees: 
#     bias: 0.12080594000000001
#     variance: 0.00525606
#     GSE: 0.126062