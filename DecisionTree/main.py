from decisiontree import DecisionTree, InformationGain, GiniGain, MajorityErrorGain

dataset_loc = "../../datasets/car/"

train_dataset = []
with open(dataset_loc + "train.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        t_dict = { # TODO: better way of doing this?
            "buying": terms[0],
            "maint": terms[1],
            "doors": terms[2],
            "persons": terms[3],
            "lug_boot": terms[4],
            "safety": terms[5],

            "label": terms[6]
        }

        train_dataset.append(t_dict)

test_dataset = []
with open(dataset_loc + "test.csv", 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        t_dict = { # TODO: better way of doing this?
            "buying": terms[0],
            "maint": terms[1],
            "doors": terms[2],
            "persons": terms[3],
            "lug_boot": terms[4],
            "safety": terms[5],

            "label": terms[6]
        }

        test_dataset.append(t_dict)

print("datasets loaded")

tree = DecisionTree(purity_function=InformationGain)
tree.makeTree(train_dataset)

test_value = {
            "buying": "low",
            "maint": "vhigh",
            "doors": "4",
            "persons": "4",
            "lug_boot": "big",
            "safety": "med"
        }

print(tree.predict(test_value))

# print(tree.printTree())