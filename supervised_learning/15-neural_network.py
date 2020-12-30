#!/usr/bin/python3
""" Neural Network """
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """that defines a neural network with one hidden layer performing
    binary classification"""

    def __init__(self, nx, nodes):
        """Instantiation method

        Args:
            nx: the number of input features
            nodes: the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # The weights vector for the hidden layer
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        # The bias for the hidden layer
        self.__b1 = np.zeros((nodes, 1))
        # The activated output for the hidden layer
        self.__A1 = 0
        # The weights vector for the output neuron
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        # The bias for the output neuron
        self.__b2 = 0
        # The activated output for the output neuron (prediction)
        self.__A2 = 0

    @property
    def W1(self):
        """ Private instance retriever """
        return self.__W1

    @property
    def b1(self):
        """ Private instance retriever """
        return self.__b1

    @property
    def A1(self):
        """ Private instance retriever """
        return self.__A1

    @property
    def W2(self):
        """ Private instance retriever """
        return self.__W2

    @property
    def b2(self):
        """ Private instance retriever """
        return self.__b2

    @property
    def A2(self):
        """ Private instance retriever """
        return self.__A2

    def forward_prop(self, X):
        """defines a neural network with one hidden layer
            performing binary classification

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                    nx is the number of input features to the neuron
                    m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
        """
        self.forward_prop(X)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            A1: output of the hidden layer
            A2: the predicted output
            alpha: the learning rate
        """
        m = A1.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(A1, dZ2.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1a = np.matmul(self.__W2.T, dZ2)
        dZ1b = A1 * (1 - A1)
        dZ1 = dZ1a * dZ1b
        dW1 = np.matmul(X, dZ1.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - alpha * db2

        self.__W1 = self.__W1 - (alpha * dW1).T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            iterations: the number of iterations to train over
            alpha: the learning rate
            verbose: boolean that defines whether or not to print
                information about the training
            graph: boolean that defines whether or not to graph information
                about the training once the training has completed
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costList = []
        stepList = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if i % step == 0 or i == iterations:
                costList.append(self.cost(Y, self.__A2))
                stepList.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.__A2)))
        if graph:
            plt.plot(stepList, costList, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
