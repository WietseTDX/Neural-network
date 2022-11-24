from Model import Model
from Layer import Layer
from Layer import Activation_Sigmoid
from Layer import Loss_Mean_Squared_error
from Dataset import load_data_from_csv

train, validation, test = load_data_from_csv("Dataset/iris_data.csv", "Dataset/iris_data_val.csv", "Dataset/iris_data_test.csv")
train_X, train_y = train
validation_X, validation_y = validation
test_X, test_y = test


model = Model()

model.add(Layer(n_inputs=4,n_neurons=8, activation=Activation_Sigmoid()))
model.add(Layer(n_inputs=8,n_neurons=3, activation=Activation_Sigmoid()))

model.set(loss=Loss_Mean_Squared_error(), learning_rate=0.05)
model.finalize()

model.train(train_X, train_y, epochs=1000, print_every=10)
print("Neural network accuracy:", model.validate(validation_X, validation_y)*100, "%")
