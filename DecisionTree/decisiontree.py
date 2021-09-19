import math
import numpy as np

class TreeNode(object):
    def __init__(self, data = None):
        self.data = data
        self.children = []

def Entropy(data: list):
    counter = {}

    for d in data:
        if counter.get(d["label"]) == None:
            counter[d["label"]] = 1
        else: 
            counter[d["label"]] += 1
    
    entropy = 0
    for v in counter.values():
        entropy += (v / len(data)) * math.log(v / len(data))

    return -entropy

def InformationGain(data: list, attribute: str):
    unique_vals = np.unique(np.array([d[attribute] for d in data]))
    gain = 0
    for val in unique_vals:
        subset = []
        for d in data:
            if d[attribute] == val:
                subset.append(d)
        gain += (len(subset) / len(data)) * Entropy(subset)
    
    return(Entropy(data) - gain)


def MajorityError(data: list):
    # TODO:
    print("majority error")

def GiniIndex(data: list):
    # TODO:
    print("gini")

def mostCommonLabel(data):
    vals, counts = np.unique(np.array([d['label'] for d in data]))
    i = np.argmax(counts)
    return vals[i]

def allSame(data):
        return len(np.unique(np.array([d['label'] for d in data]))) == 1

class DecisionTree:
    def __init__(self, purity_function = InformationGain, max_depth = None):
        self.root = TreeNode()
        self.purity_function = purity_function
        self.max_depth = 9999 if max_depth == None else max_depth
        self.mostLabel = 'na'

    # public makeTree starter function
    def makeTree(self, data: list):
        self.mostLabel = mostCommonLabel(data)
        self.root = self._makeTree(data, self.root, 0)

    # private recursive _makeTree function
    def _makeTree(self, data: list, node, depth):
        # base cases
        if len(data) == 0: # if the set of data is empty,
            return TreeNode(data=('leaf', self.mostLabel)) # return a node with the most common label
        if allSame(data): # if the data all have the same label
            return TreeNode(data=('leaf', data[0]['label'])) # return a node with that label
        if depth >= self.max_depth: # if the max depth has been met, 
            return TreeNode(data=('leaf', mostCommonLabel(data))) # return a node with the most common label

        # find best split given purity function
        min = {
            "val": 9999,
            "attr": 'attrib'
        }
        for attr in data[0].keys():
            if attr == 'label':
                continue
            purity = self.purity_function(data, attr)
            if purity < min['val']:
                min['val'] = purity
                min['attr'] = attr

        # for every unique value of the best split attribute, make a new child node
        unique_vals = np.unique(np.array([d[min['attr']] for d in data]))
        for val in unique_vals:
            childNode = TreeNode(data=('split', min['attr'], val))
            new_data = []
            for d in data:
                if d[min['attr']] == val:
                    new_val = d.copy()
                    del new_val[min['attr']]
                    new_data.append(new_val)

            node.children.append(self._makeTree(new_data, childNode, depth+1))

        return node