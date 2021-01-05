#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """Instantiation Method

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each
                layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                f = np.sqrt(2 / nx)
                self.weights['W1'] = np.random.randn(layers[i], nx) * f
            else:
                f = np.sqrt(2 / layers[i - 1])
                self.weights[W_key] = np.random.randn(layers[i],
                                                      layers[i - 1]) * f

    @property
    def L(self):
        """ property setter for the attribute """
        return self.__L

    @property
    def cache(self):
        """ property setter for the attribute """
        return self.__cache

    @property
    def weights(self):
        """ property setter for the attribute """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the
            neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
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
        A_final = self.forward_prop(X)[0]
        A_adjus = np.where(A_final >= 0.5, 1, 0)
        cost = self.cost(Y, A_final)
        return A_adjus, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            cache: dictionary containing all the intermediary
                values of the network
            alpha: the learning rate
        """
        weights = self.__weights.copy()
        m = Y.shape[1]

        for i in reversed(range(self.__L)):
            if i == self.__L - 1:
                dZ = cache['A{}'.format(i + 1)] - Y
                dW = np.matmul(cache['A{}'.format(i)], dZ.T) / m
            else:
                dZa = np.matmul(weights['W{}'.format(i + 2)].T, dZ)
                dZb = (cache['A{}'.format(i + 1)]
                       * (1 - cache['A{}'.format(i + 1)]))
                dZ = dZa * dZb

                dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m

            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i == self.__L - 1:
                self.__weights['W{}'.format(i + 1)] = \
                    (weights['W{}'.format(i + 1)]
                     - (alpha * dW).T)

            else:
                self.__weights['W{}'.format(i + 1)] = \
                    weights['W{}'.format(i + 1)] \
                    - (alpha * dW)

            self.__weights['b{}'.format(i + 1)] = \
                weights['b{}'.format(i + 1)] \
                - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: is a boolean that defines whether or not to print
              information about the training
            graph:  boolean that defines whether or not to graph information
              about the training once the training has completed.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        steps_list = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__cache['A{}'.format(self.L)])
                cost_list.append(cost)
                steps_list.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(steps_list, cost_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.savefig("23-figure")
        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Save the instance object
            to a file in pickle format
        """
        try:
            pkl = ".pkl"
            if filename[-4:] != pkl:
                filename += pkl
            with open(filename, "wb") as f:
                pickle.dump(self,
                            f,
                            pickle.HIGHEST_PROTOCOL
                            )
        except Exception:
            pass

    @staticmethod
    def load(filename):
        """
            Loads a pickled
            DeepNeuralNetwork object
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
