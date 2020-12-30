#!/usr/bin/env python3
""" Neuron """
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Neuron Class
    """

    def __init__(self, nx):
        """Instantiation method

        Args:
                nx (int): number of input features to the neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.normal(0, 1, (1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """W which stand for weight is the weight of the connection

        Returns:
                Private instance W
        """
        return self.__W

    @property
    def b(self):
        """b which stand for bias is the bias of the neuron which
            allows determines the output

        Returns:
                Private instance b
        """
        return self.__b

    @property
    def A(self):
        """A which stand for Activated output is the known output
            of the neuron (expected)

        Returns:
                Private instance A
        """
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
                X (array): numpy.ndarray with shape (nx, m) that
                    contains the input data
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1.0 / (1.0 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
                Y (array): numpy.ndarray with shape (1, m) that contains the
                        correct labels for the input data
                A (array): A is a numpy.ndarray with shape (1, m) containing
                        the activated output of the neuron for each example
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions

        Args:
            X (array): numpy.ndarray with shape (nx, m) that contains
                the input data
                    nx: number of input features to the neuron
                    m: number of examples
            Y (array): numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
        """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) that contains
                the input data
                    nx: number of input features to the neuron
                    m:  number of examples
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
            alpha: is the learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - (alpha * dW).T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron by updating the private
            attributes __W, __b, and __A

        Args:
            X: numpy.ndarray with shape (nx, m)
                that contains the input data
                    nx is the number of input features to the neuron
                    m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains
                the correct labels for the input data
            iterations:  is the number of iterations to train over
            alpha:  is the learning rate
            verbose:  boolean that defines whether or not to print
                information about the training. If True, print Cost after
                {iteration} iterations: {cost} every
                step iterations:
            graph: graph is a boolean that defines whether
                or not to graph information about the training once
                the training has completed. If True:
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise TypeError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costList = []
        stepList = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if (i % step == 0 or i == iterations):
                costList.append(self.cost(Y, self.__A))
                stepList.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(i, self.cost(Y, self.__A)))
        if graph:
            plt.plot(stepList, costList, 'b-')
            plt.plot(stepList, costList, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            # plt.show()
            plt.savefig("7-figure")
        return self.evaluate(X, Y)
