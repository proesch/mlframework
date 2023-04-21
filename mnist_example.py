import numpy as np
from api.feature_scaling import StandardScaler
from api.data_helper import create_one_hot
from api.neuralnet import sigmoid, sigmoid_derivative, forwardpass_with_softmax, backpropagation_for_softmax_with_regularization
from api.metrics import categorical_cross_entropy_with_regularization
from api.optimizers import gradient_descent_with_momentum_and_regularization
from api.training import NeuralNetTrainer
import mnist_downloader

# Importiere den MNIST-Datensatz
mnist_downloader.download_and_unzip("./mnist")
from mnist import MNIST
mndata = MNIST('mnist', return_type="numpy")

# Daten laden und splitten
all_train_Data, all_train_labels = mndata.load_training()
data_train, labels_train = all_train_Data[:42000], all_train_labels[:42000]
data_validation, labels_validation = all_train_Data[42000:], all_train_labels[42000:]
data_test, labels_test = mndata.load_testing()

# Scaling (z-Transformation)
scaler = StandardScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_validation = scaler.transform(data_validation)
data_test = scaler.transform(data_test)

# Labels konvertieren
n_categories = labels_train.max() + 1
labels_train = create_one_hot(labels_train, n_categories)
labels_validation = create_one_hot(labels_validation, n_categories)
labels_test = create_one_hot(labels_test, n_categories)

# -----------------------------------
# Training initialisieren
trainer = NeuralNetTrainer(
    trainingData=data_train,
    trainingGroundTruth=labels_train,
    validationData=data_validation,
    validationGroundTruth=labels_validation,
    activationFunction=sigmoid,
    activationDerivative=sigmoid_derivative,
    forwardpassImplementation=forwardpass_with_softmax,
    backpropImplementation=backpropagation_for_softmax_with_regularization,
    gradientDescentImplementation=gradient_descent_with_momentum_and_regularization,
    lossFunction=categorical_cross_entropy_with_regularization,
    _lambda=0.9,
    beta=0.8
)

# Netz erstellen
trainer.createLayers([
    data_train.shape[1],        # Input-Neuronen
    20,                         # 1. Hidden Layer 
    20,                         # 2. Hidden Layer
    labels_train.shape[1]       # Output
])

# Hyperparameter und Training
trainer.trainNeuralNet(
    epochs=1000, 
    alpha=0.3
)

trainer.getTestDataPerformance(data_test, labels_test)
