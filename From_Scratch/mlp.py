import numpy as np
from random import random


class MLP(object):
    

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
    
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
       
        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error, verbose=False):
     final error of the input
        

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def gradient_descent(self, learningRate=1):
        
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("original weights {}".format(weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate
            # print("Updated weights {}".format(weights))

    def train(self, inputs, targets, epochs, learningRate):

        for i in range(epochs):
            sum_error = 0
            for j, (input, target) in enumerate(zip(inputs, targets)):
                output = self.forward_propagate(input)
                error = target - output

                self.back_propagate(error)

                mlp.gradient_descent(learningRate=learningRate)

                sum_error += self.mse(target, output)
            print('Error: {} at epoch {}'.format(sum_error / len(inputs), i + 1))

    def mse(self, targets, outputs):
        return np.average((targets - outputs) ** 2)

    def _sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train
    mlp.train(inputs, targets, 50, 0.1)

#     dummy data
    input= np.array([0.1, 0.3])
    target= np.array([0.4])

    output=mlp.forward_propagate(input)
    print("According to the network {} is {}".format(target, output))
