import numpy as np
import sys
from Dataset import load_data_from_csv
from Layer import Layer
from Layer import Activation_Sigmoid
from Layer import Weights_Update
from Layer import Bias_Update
from Layer import Layer_Input
from typing import Callable, Type, Optional, List


class Model: 
    """
    The NN model
    """
    def __init__(self):
        """
        Init vars for the NN model and set the seed of randomness to 0 for a constant output
        """
        np.random.seed(0)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.layers = []

    def add(self, layer: Layer):
        """Add a layer

        Args:
            layer (Layer): A layer object to add to the model
        """        
        self.layers.append(layer)

    def set(self, *, loss: Callable, learning_rate: int):
        """Set the cost function and the optimizer function

        Args:
            loss (Callable): the loss function
            learning_rate (int): learning change rate
        """        
        self.loss = loss
        self.learning_rate = learning_rate

    def train(self, X: Optional[List[List[Optional[int]]]], y: Optional[List[List[Optional[int]]]], *, epochs=1, print_every=1) -> None:
        """Train the NN with known input and outputs for a known amount of times and print the output every x cycles

        Args:
            X (Optional[List[List[Optional[int]]]]): Features form dataset
            y (Optional[List[List[Optional[int]]]]): Labels from dataset
            epochs (int, optional): The amount of epochs to run. Defaults to 1.
            print_every (int, optional): Which interval to print the epoch. Defaults to 1.
        """                  
        for epoch in range(1, epochs+1):
            train_order = [i for i in range(len(X))]
            np.random.shuffle(train_order)
            
            sum_loss = 0
            
            for train_sample_index in train_order:
                output = self.forward(X[train_sample_index])        
                sum_loss += self.loss.calculate(output, y[train_sample_index])
                
                self.backward(y[train_sample_index])
                self.update_params()

            if not epoch % print_every:
                print(f"epoch : {epoch}, " +
                      f"loss : {sum_loss}")
               
    def validate(self, X: Optional[List[List[Optional[int]]]], y: Optional[List[List[Optional[int]]]]) -> int:
        """Validate the NN with known input and outputs without backpropagation

        Args:
            X (Optional[List[List[Optional[int]]]]): Features from dataset
            y (Optional[List[List[Optional[int]]]]): Labels form dataset

        Returns:
            int: The accuarcy
        """        
        correct_predections = 0
        total_predictions = 0

        print("\nValidation")
        for validate_sample_X, validate_sample_y in zip(X, y):
            prediction = self.forward(validate_sample_X)

            max_index_prediction = np.argmax(prediction)
            max_index_y = np.argmax(validate_sample_y)
            print("Sample:", validate_sample_X,"\tPrediction:",prediction.T, "\tActual:",validate_sample_y, "\tMax Pred:",max_index_prediction, "\tMax Actual:",max_index_y)
            
            if max_index_prediction == max_index_y:
                correct_predections+=1
            total_predictions+=1
        print("")
        if total_predictions == 0:
            return 0
        return correct_predections / total_predictions

    def update_params(self):
        """
        Update the weights and biasses
        """
        for layer in reversed(self.layers):
            layer.update_params(self.learning_rate)
    
    def backward(self, y: Optional[List[List[Optional[int]]]]) -> None:
        """Perform backpropagation with a known output

        Args:
            y (Optional[List[List[Optional[int]]]]): The labels of the dataset
        """        
        self.layers[-1].error = self.loss.backward(self.layers[-1].z, y, self.layers[-1].a)

        for i in range( len(self.layers) - 2, -1, -1): # Loop backwards, skipping the last layer due to its different error function
            layer = self.layers[i]
            layer.backward(layer.z, layer.next.error, layer.next.weights)
    
    def forward(self, X: Optional[List[List[Optional[int]]]]) -> list:
        """Perform forward propagation

        Args:
            X (Optional[List[List[Optional[int]]]]): Features of the dataset

        Returns:
            list: output of the output layer
        """
        self.input_layer.forward(X)
        
        i = 0
        for layer in self.layers:   
            layer.forward(layer.prev.a)

        return layer.a
     
    def finalize(self) -> None:
        """
        Finalize the setup of the NN by connecting all the layers.
        """
        self.input_layer = Layer_Input()
        
        layer_count = len(self.layers)
        self.trainable_layers = []
        
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]