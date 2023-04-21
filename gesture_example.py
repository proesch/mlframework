import numpy as np
from api.feature_scaling import StandardScaler
from api.neuralnet import sigmoid, sigmoid_derivative, forwardpass_with_softmax, backpropagation_for_softmax_with_regularization
from api.metrics import categorical_cross_entropy_with_regularization
from api.training import NeuralNetTrainer
from api.optimizers import gradient_descent_with_momentum_and_regularization

# Gesten-Daten laden und splitten
data_train = np.load('./gesture/X_train.npy')
labels_train = np.load('./gesture/y_train.npy')
data_validation = np.load('./gesture/X_validation.npy')
labels_validation = np.load('./gesture/y_validation.npy')

# Scaling (z-Transformation)
scaler = StandardScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_validation = scaler.transform(data_validation)

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
    30,                         # 1. Hidden Layer 
    15,                         # 2. Hidden Layer
    labels_train.shape[1]       # Output
])

# Hyperparameter und Training
trainer.trainNeuralNet(
    epochs=1000, 
    alpha=0.3
)