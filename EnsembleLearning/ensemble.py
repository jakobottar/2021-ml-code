from DecisionTree import DecisionTree

class AdaBoost:
    def __init__(self):
        self.trees = []
        self.weights = list

    def train(self, data, epochs: int=100):
        self.weights = [1/len(data)] * len(data)

        for i in range(epochs):
           stump = DecisionTree(max_depth=1) 
           stump.makeTree(data=data)
           self.trees.append(stump)
           print(stump.toJSON())