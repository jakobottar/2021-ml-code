import math
import numpy as np
from statistics import mode

class TreeNode(object):
    def __init__(self, nodetype = None, attr = None, value = None, finalclass = None):
        self.type = nodetype
        self.attr = attr
        self.value = value
        self.finalclass = finalclass
        self.children = []

    def toJSON(self):
        dict = {
            "type": self.type,
            "attr": self.attr,
            "value": self.value,
            "finalclass": self.finalclass,
            "children": []
        }

        for c in self.children:
            dict["children"].append(c.toJSON())

        return dict

def mostCommonLabel(data):
    labels = [d['label'] for d in data]
    return mode(labels)

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

def MajorityError(data: list):
    # TODO:
    label = mostCommonLabel(data)
    error = 0
    for d in data:
        if d['label'] == label:
            error += 1

    return error / len(data)

def GiniIndex(data: list):
    counter = {}

    for d in data:
        if counter.get(d["label"]) == None:
            counter[d["label"]] = 1
        else: 
            counter[d["label"]] += 1
    
    gini = 0
    for v in counter.values():
        gini += (v / len(data))**2

    return 1 - gini

def InformationGain(data: list, attribute: str, purity = Entropy):
    unique_vals = np.unique(np.array([d[attribute] for d in data]))
    gain = 0
    for val in unique_vals:
        subset = []
        for d in data:
            if d[attribute] == val:
                subset.append(d)
        gain += (len(subset) / len(data)) * purity(subset)
    
    return(purity(data) - gain)

def GiniGain(data: list, attribute: str):
    return InformationGain(data, attribute, GiniIndex)

def MajorityErrorGain(data: list, attribute: str):
    return InformationGain(data, attribute, MajorityError)

def allSame(data):
        return len(np.unique(np.array([d['label'] for d in data]))) == 1

class DecisionTree:
    def __init__(self, purity_function = InformationGain, max_depth = None):
        self.root = TreeNode(nodetype="root")
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
            node.type = 'leaf'
            node.finalclass = self.mostLabel
            return node # return a node with the most common label
        if allSame(data): # if the data all have the same label
            node.type = 'leaf'
            node.finalclass = data[0]['label']
            return node # # return a node with that label
        if depth >= self.max_depth: # if the max depth has been met, 
            node.type = 'leaf'
            node.finalclass = mostCommonLabel(data)
            return node # return a node with the most common label

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
            childNode = TreeNode(nodetype='split', attr=min['attr'], value=val)
            new_data = []
            for d in data:
                if d[min['attr']] == val:
                    new_val = d.copy()
                    del new_val[min['attr']]
                    new_data.append(new_val)

            node.children.append(self._makeTree(new_data, childNode, depth+1))

        return node
    
    # prints tree in JSON format
    def printTree(self):
        #TODO: 
        return self.root.toJSON()

    # predicts label based on attributes
    def predict(self, value):
        #TODO: 
        return self._predict(value, self.root)

    def _predict(self, value, node):
        if node.type == 'leaf':
            return node.finalclass
        
        for child in node.children:
            attr = child.attr
            if value[attr] == child.value:
                return self._predict(value, child)
