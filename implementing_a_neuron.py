import numpy as np


def activate(inputs, weights):
    h = 0

    for i in range(len(inputs)):
        h += inputs[i] * weights[i]
    return 1 / (1 + np.exp(h))


inputs = [1, 2, 3]
weights = [.1, .4, .5]

output = activate(inputs, weights)
print(output)
