from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
import gradient

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "../data/concrete/"

cc_train_x = []
cc_train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_train_x.append(terms_flt[:-1])
        cc_train_y.append(terms_flt[-1])

cc_test_x = []
cc_test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_test_x.append(terms_flt[:-1])
        cc_test_y.append(terms_flt[-1])

### ============================================================================================
print("LMS with Batch Gradient Descent")
bgd, loss_bgd = gradient.BatchGradientDescent(cc_train_x, cc_train_y, lr = 1e-3, epochs = 500)

print(f"weight vect: {bgd}")

# pred_train = bgd.predict(cc_train_x)
# print(f"training MSE: {gradient.MSE(pred_train, cc_train_y)}")

# pred_test = bgd.predict(cc_test_x)
# print(f"testing MSE: {gradient.MSE(pred_test, cc_test_y)}")

### ============================================================================================
print("LMS with Stochastic Gradient Descent")
sgd, loss_sgd = gradient.StochasticGradientDescent(cc_train_x, cc_train_y, lr = 1e-3, epochs = 500)

print(f"weight vect: {sgd}")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.array(range(len(loss_bgd))) * len(cc_train_x), loss_bgd, color = 'tab:blue', label = "batch")
ax.plot(loss_sgd, color = 'tab:orange', label = "stochastic")
ax.legend()
ax.set_title("Gradient Descent")
ax.set_xlabel("# of iterations")
ax.set_ylabel("Mean Squared Error")

plt.savefig("./out/gd_error.png")
plt.clf()

# pred_train = bgd.predict(cc_train_x)
# print(f"training MSE: {gradient.MSE(pred_train, cc_train_y)}")

# pred_test = bgd.predict(cc_test_x)
# print(f"testing MSE: {gradient.MSE(pred_test, cc_test_y)}")

### ============================================================================================
print("LMS Analytic Method")
lms = gradient.LMSRegression(cc_train_x, cc_train_y)

print(f"weight vect: {lms}")

# pred_train = lms.predict(cc_train_x)
# print(f"training MSE: {gradient.MSE(pred_train, cc_train_y)}")

# pred_test = lms.predict(cc_test_x)
# print(f"testing MSE: {gradient.MSE(pred_test, cc_test_y)}")