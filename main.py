import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[0:10000]
y_train = y_train[0:10000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]
print("Training Data: ", x_train.shape)

# Features Extraction Using Centroid
def grids(image, row, column):
    x, y = image.shape
    if x % row != 0:
        print("Rows are not Evenly Divisible by ", x, row)
    if y % column != 0:
        print("Columns are not Evenly Divisible by ", y, column)
    return image.reshape(x // row, row, -1, column).swapaxes(1, 2).reshape(-1, row, column)


# Features Extraction Using Centroid
def centroid(image):
    featureVector = []
    for grid in grids(image, 2, 2):
        Xc = 0
        Yc = 0
        sum = 0
        for i, x in np.ndenumerate(grid):
            sum += x
            Xc += x * i[0]
            Yc += x * i[1]
        if sum != 0:
            featureVector.append(Xc / sum)
            featureVector.append(Yc / sum)
        else:
            featureVector.append(0)
            featureVector.append(0)
    return np.array(featureVector)


# **********************************************************************************************************************

NodesNumber = 6
LayerNumber = 1
LearningRate = 0.1

# **********************************************************************************************************************

print("Nodes Number : " + str(NodesNumber))
print("Layer Number : " + str(LayerNumber))
print("Learning Rate : " + str(LearningRate))

# Making the Train Features
print("Calculating Features ...")
trainFeatures = []
for image in x_train:
    trainFeatures.append(centroid(image))
trainFeatures = np.array(trainFeatures)

x, y = trainFeatures.shape

# initializing Weights and bias
Weights = np.random.rand(LayerNumber, NodesNumber, y)
NodeValues = np.random.rand(LayerNumber, NodesNumber)
OutputWeights = np.random.rand(10, NodesNumber)
OutputValues = np.random.rand(10)
Delta = np.random.rand(LayerNumber, 10)

# Creating one hot result vector
ResultFeatures = []
lr = np.arange(10)
for result in y_train:
    one_hot = (lr == result).astype(int)
    ResultFeatures.append(one_hot)
ResultFeatures = np.array(ResultFeatures)


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Maxindex(x):
    temp = 0
    for values in x:
        if temp < values:
            temp = values
    for index in range(10):
        if temp == x[index]:
            return index


CorrectGuess = 0
print("Training Network ...")
for image in range(x):
    for epoch in range(100):

        # Hidden Layer
        for Layer in range(LayerNumber):
            for Feature in range(y):
                for Nodes in range(NodesNumber):
                    NodeValues[Layer][Nodes] = NodeValues[Layer][Nodes] + (
                            trainFeatures[image][Feature] * Weights[Layer][Nodes][Feature])
            for Nodes in range(NodesNumber):
                NodeValues[Layer][Nodes] = sigmoid(NodeValues[Layer][Nodes])

        # Output Layer
        for HiddenL in range(NodesNumber):
            for Nodes in range(10):
                OutputValues[Nodes] = OutputValues[Nodes] + (NodeValues[Layer][HiddenL] * OutputWeights[Nodes][HiddenL])
        for Nodes in range(10):
            OutputValues[Nodes] = sigmoid(OutputValues[Nodes])

        # **********************************************************************************************************************

        # Back Propagation
        error = 0
        for Results in range(10):
            error = (pow((ResultFeatures[image][Results] - OutputValues[Results]), 2)) + error
            Delta[Layer][Results] = (ResultFeatures[image][Results] - OutputValues[Results]) * OutputValues[Results] * (
                    1 - OutputValues[Results])

        # if error < 0.1:
        #     break

        # Output Layer
        for HiddenL in range(NodesNumber):
            for Nodes in range(10):
                OutputWeights[Nodes][HiddenL] = OutputWeights[Nodes][HiddenL] + (
                        Delta[Layer][Nodes] * NodeValues[Layer][HiddenL] * LearningRate)

        # Hidden Layers
        for Layer in range(LayerNumber,0):
            for Feature in range(y):
                for HiddenL in range(NodesNumber):
                    temp = 0
                    for Nodes in range(NodesNumber):
                        temp = temp + (Delta[Layer][Nodes] * Weights[Nodes][HiddenL])
                    Weights[Layer][HiddenL][Feature] = Weights[Layer][HiddenL][Feature] + (trainFeatures[image][Feature] *
                                                        (1 - NodeValues[Layer][HiddenL]) *
                                                        NodeValues[Layer][HiddenL]
                                                        * temp * LearningRate)
                    if Layer - 1 != -1:
                        Delta[Layer - 1][Nodes] = (1 - NodeValues[Layer][HiddenL]) * NodeValues[Layer][HiddenL] * temp

    print(ResultFeatures[image])
    print(OutputValues)

    if y_train[image] == Maxindex(OutputValues):
        CorrectGuess += 1
    print(str(CorrectGuess) + " / " + str(image + 1))

print("Training Accuracy = ", end="")
print((CorrectGuess / x) * 100, end="")
print("%")
