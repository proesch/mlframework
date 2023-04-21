import numpy as np
from .data_helper import saveThetas, create_one_hot
from .neuralnet import generate_random_thetas
from .metrics import categorical_cross_entropy_with_regularization, getAccuracy, f1_scores_mean, classfiyHypothesis

class NeuralNetTrainer:
    def __init__(self, trainingData, trainingGroundTruth, validationData, validationGroundTruth, activationFunction, activationDerivative, forwardpassImplementation, backpropImplementation, gradientDescentImplementation, lossFunction, _lambda=None, beta=None):
        self.trainingData = trainingData
        self.trainingGroundTruth = trainingGroundTruth
        self.validationData = validationData
        self.validationGroundTruth = validationGroundTruth
        self.activationFunction = activationFunction
        self.activationDerivative = activationDerivative
        self.forwardpassImplementation = forwardpassImplementation
        self.backpropImplementation = backpropImplementation
        self.gradientDescentImplementation = gradientDescentImplementation
        self.lossFunction = lossFunction
        self._lambda = _lambda
        self.beta = beta

        self.min_J_validation = 999
        self.epoch = 0
        self.thetas = []

    def createLayers(self, n_neurons_per_layer):
        for layer in range(len(n_neurons_per_layer)-1):
            self.thetas.append(generate_random_thetas(n_neurons_per_layer[layer+1], n_neurons_per_layer[layer]))

    def trainNeuralNet(self, epochs, alpha, name="default_training", path="./"):
        error_history = np.ones((epochs+1, 2)) # [0] ist train, [1] ist validation
        accuracy_history = np.ones((epochs+1, 2)) # [0] ist train, [1] ist validation
        f1_history = np.ones((epochs+1, 2))   # [0] ist train, [1] ist validation
        
        while self.epoch <= epochs:
            J_train, accuracy_train, f1_train, J_validation, accuracy_validation, f1_validation = self.trainSingleEpoch(self.epoch, alpha, name, path)
            error_history[self.epoch, 0] = J_train
            accuracy_history[self.epoch, 0] = accuracy_train
            f1_history[self.epoch, 0] = f1_train
            error_history[self.epoch, 1] = J_validation
            accuracy_history[self.epoch, 1] = accuracy_validation
            f1_history[self.epoch, 1] = f1_validation
            self.epoch += 1

        return error_history, accuracy_history, f1_history

    def trainSingleEpoch(self, epoch, alpha, name, path):
        # Training Data
        activations_train, linearComponents_train, J_train, accuracy_train, f1_train = self.forwardPassWithMetrics(self.trainingData, self.trainingGroundTruth)
        
        # Validation Data
        activations_validation, linearComponents_validation, J_validation, accuracy_validation, f1_validation = self.forwardPassWithMetrics(self.validationData, self.validationGroundTruth)

        self.printCurrentTrainingResult(epoch, J_train, accuracy_train, f1_train, J_validation, accuracy_validation, f1_validation)
        if(J_validation < self.min_J_validation):
            self.min_J_validation = J_validation
            saveThetas(self.thetas, epoch, path + name)
      
        backpropArguments = [self.trainingData, self.trainingGroundTruth, self.thetas, activations_train, linearComponents_train, self.activationDerivative]
        if self._lambda != None:
            backpropArguments.append(self._lambda)
        bp = self.backpropImplementation(*backpropArguments)

        gradientDescentArguments = [bp, self.thetas, alpha]
        if self.beta != None:
            gradientDescentArguments.append(self.beta)
        if self._lambda != None:
            gradientDescentArguments.append(self._lambda)
        self.gradientDescentImplementation(*gradientDescentArguments)
        
        return J_train, accuracy_train, f1_train, J_validation, accuracy_validation, f1_validation

    def printCurrentTrainingResult(self, epoch, trainingLoss, trainingAccuracy, trainingF1, validationLoss, validationAccuracy, validationF1):
        print("Epoch: ", epoch, "Val. Loss: ", validationLoss, "Val. F1: ", validationF1, "Train. Loss: ", trainingLoss, "Train F1:", trainingF1)

    def forwardPassWithMetrics(self, data, groundTruth):
        activations, linearComponents = self.forwardpassImplementation(data, self.thetas, self.activationFunction)
        
        lossFunctionArguments = [activations[-1], groundTruth]
        if self._lambda != None:
            lossFunctionArguments.extend([self._lambda, self.thetas])
        J = self.lossFunction(*lossFunctionArguments)

        accuracy = getAccuracy(activations[-1], groundTruth)
        h_one_hot = create_one_hot(classfiyHypothesis(activations[-1]), groundTruth.shape[1])
        f1 = f1_scores_mean(h_one_hot, groundTruth)
        return activations, linearComponents, J, accuracy, f1

    def getTestDataPerformance(self, testData, testGroundTruth):
        activations_test, linearComponents_test, J_test, accuracy_test, f1_test = self.forwardPassWithMetrics(testData, testGroundTruth)
        print("Test Data Loss: ", J_test, "F1:", f1_test, "Acc.: ", accuracy_test)