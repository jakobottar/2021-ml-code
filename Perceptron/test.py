import numpy as np

#create NumPy array
my_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#view NumPy array
print(my_array)

new_array = np.insert(my_array, 0, [1]*len(my_array), axis=1)

print(new_array)