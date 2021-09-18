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
    # TODO:
    unique_vals = np.unique(np.array([d[attribute] for d in data]))
    gain = 0
    for val in unique_vals:
        subset = []
        for d in data:
            if d[attribute] == val:
                subset.append(d)
        gain += (len(subset) / len(data)) * Entropy(subset)
    
    print(Entropy(data) - gain)
    return(Entropy(data) - gain)


def MajorityError(data: list):
    # TODO:
    print("majority error")

def GiniIndex(data: list):
    # TODO:
    print("gini")

class DecisionTree:
    def __init__(self, purity_function = InformationGain):
        self.root = TreeNode()
        self.purity_function = purity_function

        print("a tree!")

    def allSame(self, data):
        return len(np.unique(np.array([d['label'] for d in data]))) == 1

    def makeTree(self, data: list):
        self.root = self._makeTree(data, self.root)


    def _makeTree(self, data: list, node):
        """
        if data is all same label:
            leaf node, label all as label
        
        find attrib a that best splits dataset
        for each value of a, create new branch and call again
        """
        if len(data) == 0:
            return TreeNode(data=('leaf', 'na'))
        if self.allSame(data):
            return TreeNode(data=('leaf', data[0]['label']))

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
        
        # data_new = data.copy()
        # for d in data_new:
        #     del d[min['attr']]

        for val in unique_vals:
            childNode = TreeNode(data=('split', min['attr'], val))
            new_data = []
            for d in data:
                if d[min['attr']] == val:
                    new_val = d.copy()
                    del new_val[min['attr']]
                    new_data.append(new_val)

            self._makeTree(new_data, childNode)
            node.children.append(childNode)

        return node