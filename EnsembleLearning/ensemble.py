from DecisionTree import DecisionTree

class AdaBoost:
    def __init__(self):
        self.trees = []
        self.weights = []

    def train(self, data, epochs: int=100):
        self.weights = [1/len(data)] * len(data)

        for i in range(epochs):
            ## 1: generate stump given weights
            stump = DecisionTree() 
            stump.makeTree(data=data, weights=self.weights, max_depth=1)
            self.trees.append(stump)
            print(stump.toJSON())
            ## 2: calculate error
            ## 3: calculate say
            ## 4: update weights