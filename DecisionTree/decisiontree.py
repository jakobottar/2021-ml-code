import math

class NTree(object):
    def __init__(self):
        self.data = None
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

def InformationGain(data: list):
    # TODO:
    print("info gain")

def MajorityError(data: list):
    # TODO:
    print("majority error")

def GiniIndex(data: list):
    # TODO:
    print("gini")

class DecisionTree:
    def __init__(self, purity_function: function = InformationGain):
        self.root = NTree()
        self.purity_function = purity_function

        print("a tree!")

    def makeTree(self, data: list):
        Entropy(data)

    def _makeTree(self, data: list):
        """
        if data is all same label:
            leaf node, label all as label
        
        find attrib a that best splits dataset
        for each value of a, create new branch and call again
        """