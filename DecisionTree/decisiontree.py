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


def allSame(data):
        return len(np.unique(np.array([d['label'] for d in data]))) == 1

class DecisionTree:
    def __init__(self, purity_function = InformationGain, max_depth = None):
        self.root = TreeNode()
        self.purity_function = purity_function
        self.max_depth = 9999 if max_depth == None else max_depth

        print("a tree!")

    def makeTree(self, data: list):
        self.root = self._makeTree(data, self.root, 0)


    def _makeTree(self, data: list, node, depth):
        """
        if data is all same label:
            leaf node, label all as label
        
        find attrib a that best splits dataset
        for each value of a, create new branch and call again
        """

        if len(data) == 0:
            return TreeNode(data=('leaf', 'na'))
        if allSame(data):
            return TreeNode(data=('leaf', data[0]['label']))
        
        if depth >= self.max_depth:
            print("oops") #TODO: 

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