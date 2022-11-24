import numpy as np
import math
from typing import Callable, Type

def sigmoid(x: int) -> int:
    """Calculate sigmoid function

    Args:
        x (int): Number to take sigmoid

    Returns:
        int: The sigmoid
    """    
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x: int) -> int:
    """derivative of the sigmoid

    Args:
        x (int): Number to take the Derivative of sigmoid

    Returns:
        int: The sigmoid derivative
    """    
    return sigmoid(x) * (1 - sigmoid(x))

class Layer:
    """
    This class is a layer withing the NN
    """
    def __init__(self,*, n_inputs: int, n_neurons: int, activation: Callable) -> None:
        """Initialization that is necessary for a layer within the NN, set the amount of inputs and neurons and the desired activation function

        Args:
            n_inputs (int): The inputs from the last layer
            n_neurons (int): How many neurons in this layer
            activation (Callable): The activation function
        """        
        self.weights = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.biasses = np.zeros((1,n_neurons)).T
        self.activation = activation
        self.weights_update_func = Weights_Update()
        self.bias_update_func = Bias_Update()

    def forward(self, inputs: list) -> None:
        """The function to perform a forward propagation on the whole layer. This function needs to be called after the layer before is called

        Args:
            inputs (list): List with input values
        """        
        self.z = np.dot(self.weights, inputs) + self.biasses
        self.a = self.activation.forward(self.z)

    def backward(self, zs_current_layer: int, deltas_next_layer: Type["Layer"], weights_to_next_layer: Type["Layer"]):
        """The function to perform backpropagation. This needs to be called form output to input layer and it cannot be called on the output layer

        Args:
            zs_current_layer (int): The current layer ZS
            deltas_next_layer (Layer): The delta next layer
            weights_to_next_layer (Layer): The weights of the next layer
        """        
        self.error = (sigmoid_der(zs_current_layer) * np.dot(deltas_next_layer.T, weights_to_next_layer).T)

    def update_params(self, learning_rate: int):
        """Update the weights and biasses

        Args:
            learning_rate (int): The accepted learning rate
        """        
        self.weights_update_func.backward(self.weights, learning_rate, self.error, self.prev.a)
        self.weights = self.weights_update_func.output

        self.bias_update_func.backward(learning_rate, self.error, self.biasses)
        self.biasses = self.bias_update_func.output


class Layer_Input:
    """
    Class te represent a input layer
    """
    def forward(self, inputs: list):
        """To introduce a value in the forward propagation

        Args:
            inputs (list): The input to take layer input
        """        
        self.a = np.array([inputs]).T        



class Activation_Sigmoid:
    """
    Perform the sigmoid function
    """
    def forward(self, inputs: int) -> int:
        """Perform the sigmoid function for forward propagation

        Args:
            inputs (int): The input for sigmoid function

        Returns:
            int: Sigmoid function
        """        
        self.output = sigmoid(inputs)
        return self.output

    def backward(self, zs_current_layer: list, deltas_next_layer: list, weights_to_next_layer: list):
        """Calculate the delta for the backpropogation

        Args:
            zs_current_layer (list): The current layer z
            deltas_next_layer (list): The next layer deltas from expected
            weights_to_next_layer (list): The weights of the next layer
        """        
        self.output = sigmoid_der(zs_current_layer) * (deltas_next_layer * weights_to_next_layer)


class Weights_Update:
    """
    Update the weights    
    """
    def backward(self, weights_to_layer: int, learning_rate: int, deltas_layer: list, a_prev_layer: list):
        """Update the weight in a back propagation

        Args:
            weights_to_layer (int): Weights to last layer
            learning_rate (int): The accepted change rate
            deltas_layer (list): The delta from expected
            a_prev_layer (list): The a of last layer
        """        
        self.output = weights_to_layer + (np.multiply(learning_rate, np.multiply(deltas_layer, a_prev_layer.T)))


class Bias_Update:
    """
    Update the bias
    """
    def backward(self, learning_rate: int, deltas_layer: list, biasses_layer: list) -> None:
        """Update the bias in a back propagation

        Args:
            learning_rate (int): The accepted change rate
            deltas_layer (list): The difference between expected
            biasses_layer (list): The current bias
        """        
        self.output = biasses_layer + np.multiply(learning_rate,deltas_layer)
        

class Loss_Mean_Squared_error:
    """
    The cost function
    """
    def calculate(self, y_pred: list, y_true: list) -> list:
        """Calculate the cost function

        Args:
            y_pred (list): The y prediction
            y_true (list): The y test set

        Returns:
            list: The losses
        """        
        sample_losses = np.square(np.subtract(y_true,np.array(y_pred).T)).mean()
        return sample_losses
    
    def backward(self, z_j: list, expected_output: list, a_o: list) -> int:
        """Calculate the error of the last layer for back propagation

        Args:
            z_j (list): The Z
            expected_output (list): The expected output, test set
            a_o (list): The A value

        Returns:
            int: _description_
        """        
        self.error = (sigmoid_der(z_j).T * (expected_output - a_o.T)).T
        return self.error