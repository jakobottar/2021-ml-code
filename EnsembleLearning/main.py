from os import makedirs
import ensemble

# try: makedirs("./out/")
# except FileExistsError: None

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

ada = ensemble.AdaBoost()
ada.train(train_dataset, 1)