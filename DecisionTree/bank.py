import argparse
from DecisionTree import DecisionTree, InformationGain, GiniGain, MajorityErrorGain

parser = argparse.ArgumentParser()
parser.add_argument("--unknown", type=bool, default=False)
FLAGS, unparsed = parser.parse_known_args()

dataset_loc = "../data/bank/"

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

train_dataset = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        train_dataset.append(HandleLine(line))

test_dataset = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        test_dataset.append(HandleLine(line))

print("datasets loaded")

tree = DecisionTree()
tree.makeTree(train_dataset)

for depth in [1,2,4,6,8,10,12,14,16]:
    for name, fun in zip(["Entropy", "Gini Index", "Majority Error"], [InformationGain, GiniGain, MajorityErrorGain]):
        print(f"training a tree with depth {depth} using purity measure {name}")
        tree = DecisionTree(purity_function=fun, max_depth=depth)
        tree.makeTree(train_dataset, handle_unknown=FLAGS.unknown)

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