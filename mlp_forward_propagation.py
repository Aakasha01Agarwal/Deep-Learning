import numpy as np


class MLP:

    def __init__(self, n_inputs=3, n_hidden_layers=None, n_output=2):
        if n_hidden_layers is None:
            n_hidden_layers = [3, 5]
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.n_output = n_output
        layers = [self.n_inputs] + n_hidden_layers + [self.n_output]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

    def forward_propagation(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate the net input
            net_input = np.dot(activations, w)

            # calculate the activations
            activations = self.sigmoid(net_input)

        return activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # create MLP
    mlp = MLP()

    #     create inputs
    inputs = np.random.rand(mlp.n_inputs)
    # perform forward
    outputs = mlp.forward_propagation(inputs)
    # print
    print("Network Inputs is ", inputs)
    print("Network Output is ", outputs)
