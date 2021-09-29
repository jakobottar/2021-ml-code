from os import makedirs
from DecisionTree import DecisionTree, InformationGain, GiniGain, MajorityErrorGain
import json

dataset_loc = "../data/car/"

def HandleLine(line):
    terms = line.strip().split(",")
    t_dict = { # TODO: better way of doing this?
        "buying": terms[0],
        "maint": terms[1],
        "doors": terms[2],
        "persons": terms[3],
        "lug_boot": terms[4],
        "safety": terms[5],
        "label": terms[6]
    }
    return t_dict

train_dataset = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        train_dataset.append(HandleLine(line))

test_dataset = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        test_dataset.append(HandleLine(line))

print("datasets loaded")

ex_tree = DecisionTree(max_depth=5)
ex_tree.makeTree(train_dataset)

try: makedirs("./out/")
except FileExistsError: None

print("exporting example tree to ./out/tree.json")
with open("out/tree.json", "w") as f:
    json.dump(ex_tree.toJSON(), f, sort_keys=True, indent=2)

for depth in range(1, 7):
    for name, fun in zip(["Entropy", "Gini Index", "Majority Error"], [InformationGain, GiniGain, MajorityErrorGain]):
        print(f"training a tree with depth {depth} using heuristic {name}")
        tree = DecisionTree(purity_function=fun, max_depth=depth)
        tree.makeTree(train_dataset)

        train_error = 0
        for val in train_dataset:
            if val["label"] != tree.predict(val):
                train_error += 1
        train_error /= len(train_dataset)

        test_error = 0
        for val in test_dataset:
            if val["label"] != tree.predict(val):
                test_error += 1
        test_error /= len(test_dataset)
        # print(f"{depth} & {name} & {'{0:.3g}'.format(train_error)} & {'{0:.3g}'.format(test_error)} \\\\")
        print(f"training error: {'{0:.3g}'.format(train_error)}, testing error: {'{0:.3g}'.format(test_error)}")
