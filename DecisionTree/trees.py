from os import makedirs
import DecisionTree
import numpy as np
import json
import math

dataset_loc = "./data/"

try: makedirs("./out/")
except FileExistsError: None

# def MakeWeights(data: list):
#     for d in [data:
#         for key, val in d.items():
#             if val == 'missing': d[key] = 

class SimpleTree:
    def __init__(self, heuristic=DecisionTree.InformationGain):
        self.root = DecisionTree.TreeNode(nodetype="root")
        self.heuristic = heuristic
    
    def MakeTree(self, data:list):
        print("let's grow a tree!")
        self._MakeTree(data, self.root, ['label'])
    
    def _MakeTree(self, data: list, node, used_attrs: list):
        if len(data) == 0: # if the set of data is empty,
            print("the node was given no data")
            node.type = "leaf"
            node.finalclass = "na"
            return node # return a node with the most common label
        if DecisionTree.allSame(data): # if the data all have the same label
            print("the node's labels are pure, make it a leaf node")
            node.type = "leaf"
            node.finalclass = data[0]["label"]
            return node # # return a node with that label

        max = { "val": -np.inf, "attr": "none_found" }
        for attr in data[0].keys():
            if attr in used_attrs:
                continue
            gain = self.heuristic(data, attr)
            print(f"gain({attr}) = {gain}")
            if gain > max["val"]:
                max["val"] = gain
                max["attr"] = attr
        
        print(f"best attribute was {max['attr']}, split on it")
        new_attrs = used_attrs.copy()
        new_attrs.append(max["attr"])

        unique_vals = np.unique(np.array([d[max["attr"]] for d in data]))
        for val in unique_vals:
            childNode = DecisionTree.TreeNode(nodetype="split", attr=max["attr"], value=val)
            new_data = [d for d in data if d[max["attr"]] == val]
            print(f"new dataset for {max['attr']} = {val}: {new_data}")
            node.children.append(self._MakeTree(new_data, childNode, new_attrs))

        return node

    def toJSON(self):
        return self.root.toJSON()

def HandleLine_p1(line):
    terms = line.strip().split(',')
    t_dict = { # TODO: better way of doing this?
        "x1": terms[0],
        "x2": terms[1],
        "x3": terms[2],
        "x4": terms[3],
        "label": terms[4]
    }
    return t_dict

def HandleLine_p2(line):
    terms = line.strip().split(',')
    t_dict = { # TODO: better way of doing this?
        "outlook": terms[0],
        "temp": terms[1],
        "humidity": terms[2],
        "wind": terms[3],
        "label": terms[4]
    }
    return t_dict

### Problem 1 =========================================================
print("PROBLEM 1 ---------------------")
problem1 = []
with open(dataset_loc + "problem1.csv", 'r') as f:
    for line in f:
        problem1.append(HandleLine_p1(line))

p1_tree = SimpleTree()
p1_tree.MakeTree(problem1)

print("exporting example tree to ./out/p1_tree.json")
with open("out/p1_tree.json", "w") as f:
    json.dump(p1_tree.toJSON(), f, sort_keys=True, indent=2)

problem1 = []
with open(dataset_loc + "problem1.csv", 'r') as f:
    for line in f:
        problem1.append(HandleLine_p1(line))

### Problem 2 =========================================================
print("PROBLEM 2 ---------------------")
problem2 = []
with open(dataset_loc + "problem2.csv", 'r') as f:
    for line in f:
        problem2.append(HandleLine_p2(line))

print("using Entropy Informaton Gain")
p2_tree_ent = SimpleTree(heuristic=DecisionTree.InformationGain)
p2_tree_ent.MakeTree(problem2)

print("using Majority Error")
p2_tree_me = SimpleTree(heuristic=DecisionTree.MajorityErrorGain)
p2_tree_me.MakeTree(problem2)

print("using Gini Index")
p2_tree_gi = SimpleTree(heuristic=DecisionTree.GiniGain)
p2_tree_gi.MakeTree(problem2)

### Problem 3 =========================================================
def ldCopy(data: list):
    newData = []
    for d in data:
        newD = d.copy()
        newData.append(newD)
    return newData

problem3 = []
with open(dataset_loc + "problem3.csv", 'r') as f:
    for line in f:
        problem3.append(HandleLine_p2(line))
### a)
print("PROBLEM 3 a --------------------")
problem3a = ldCopy(problem3)
problem3a[-1]['outlook'] = DecisionTree.mostCommon(problem3a, 'outlook')
print(problem3a[-1])

print(f"gain(outlook) = {DecisionTree.InformationGain(problem3a, 'outlook')}")
print(f"gain(temp) = {DecisionTree.InformationGain(problem3a, 'temp')}")
print(f"gain(humidity) = {DecisionTree.InformationGain(problem3a, 'humidity')}")
print(f"gain(wind) = {DecisionTree.InformationGain(problem3a, 'wind')}")

### b)
print("PROBLEM 3 b --------------------")
problem3b = ldCopy(problem3)
filtered = [x for x in problem3b if x['label'] == 'yes']
problem3b[-1]['outlook'] = DecisionTree.mostCommon(filtered, 'outlook')
print(problem3b[-1])

print(f"gain(outlook) = {DecisionTree.InformationGain(problem3b, 'outlook')}")
print(f"gain(temp) = {DecisionTree.InformationGain(problem3b, 'temp')}")
print(f"gain(humidity) = {DecisionTree.InformationGain(problem3b, 'humidity')}")
print(f"gain(wind) = {DecisionTree.InformationGain(problem3b, 'wind')}")

### c)
print("PROBLEM 3 c --------------------")
def WeightedEntropy(data: list):
    counter = {}

    for d in data:
        if counter.get(d["label"]) == None: counter[d["label"]] = d['weight']
        else: counter[d["label"]] += d['weight']
    
    entropy = 0
    for v in counter.values():
        entropy += (v / np.sum([d['weight'] for d in data]) ) * math.log(v / np.sum([d['weight'] for d in data]) )

    return -entropy

def WeightedInformationGain(data: list, attribute: str):
    gain = 0
    if type(data[0][attribute]) == str:
        unique_vals = np.unique(np.array([d[attribute] for d in data]))
        for val in unique_vals:
            subset = []
            for d in data:
                if d[attribute] == val:
                    subset.append(d)
            gain += (np.sum([d['weight'] for d in subset]) / np.sum([d['weight'] for d in data])) * WeightedEntropy(subset)