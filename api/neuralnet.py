import numpy as np

# --------------------------
# Hilfsfunktionen für Thetas
# --------------------------
# Input immer Shape (Anz Samples, Anz Inputs) - Bias wird auf Dimension 1 hinzugefügt
def addBias(inp):
    return np.insert(inp, 0, 1, 1)

# Input immer Shape (Anz Samples, Anz Inputs) - Bias wird auf Dimension 1 entfernt
def removeBias(inp):
    return inp[:, 1:]

# Thetas mit Zufallswerten initialisieren
def generate_random_thetas(n_neurons, n_inputs_per_neuron, interval=1):
    thetas = []
    for neuron in range(n_neurons):
        thetas_neuron = [1] # Bias vorausgefüllt
        for input in range(n_inputs_per_neuron):
            thetas_neuron.append(interval * (np.random.rand() - 0.5)) # Zufallswerte zwischen Interval * (-0.5 und 0.5)
        thetas.append(thetas_neuron)
    return np.array(thetas)

# ----------------------------------
# Aktivierungsfunktionen und Softmax
# ----------------------------------

# Sigmoid
def sigmoid(z):
    sig = np.where(z<0, np.exp(z)/(1+np.exp(z)), 1/(1+np.exp(-z)))
    return sig
    #return 1 / (1+np.e**-np.clip(z,-100, 100))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Tangens Hyperbolicus
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

# ReLU 
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

#Softmax
def softmax(o):
    return np.exp(o) / np.expand_dims(np.sum(np.exp(o), axis=1), 1)

# ------------
# Forward-Pass
# ------------
# Gewöhnlicher Forward-Propagation: Eingabe-Samples (X), Array mit Thetas aller Layer (thetas), Aktivierungsfunktion (activation)
def forwardpass(X, thetas, activation_function):
    input = X
    activations = []
    linearComponents = []
    for thetas_current_layer in thetas:
        z = addBias(input) @ thetas_current_layer.T
        linearComponents.append(z)
        a = activation_function(z)
        activations.append(a)
        input = a
    return activations, linearComponents

# Forwardpropagation mit Softmax in letzter Schicht: Eingabe-Samples (X), Array mit Thetas aller Layer (thetas), Aktivierungsfunktion (activation)
def forwardpass_with_softmax(X, thetas, activation_function):
    input = X
    activations = []
    linearComponents = []
    for index, thetas_current_layer in enumerate(thetas):
        z = addBias(input) @ thetas_current_layer.T
        linearComponents.append(z)
        if index == len(thetas) - 1: # keine Aktivierungsfunktion in Output-Layer, dafür Softmax
            a = softmax(z)
        else:
            a = activation_function(z)
        activations.append(a)
        input = a
    return activations, linearComponents

# ---------------
# Backpropagation
# ---------------
def backpropagation(X, y, thetas, activations, linear_components, activation_derivative):
    # Gradienten für Output-Layer
    dJ_do = -y * 1/activations[-1] + (1 - y) / (1 - activations[-1])
    do_dz = activation_derivative(linear_components[-1])
    multiplikator_D =  dJ_do * do_dz # (4, 3) - Anz_Samples, Anz_Neur_D
    d_theta_C= np.expand_dims(addBias(activations[-2]), 1) * np.expand_dims(multiplikator_D, 2) # shape (4, 3, 3) Anz_Samples, Anz_Neur_

    # Gradienten für zweiten Layer (C)
    daC_dzC = activation_derivative(activations[-2]) # aC - np.square(aC) # (4, 2) - 4 samples, 2 Neuronen auf Layer C
    dzD_daC = thetas[-1][:, 1:] # Bias-Gewicht entfernen # (3, 2) - 3 Neuronen Layer D, mit je 2 thetas (Anz. Neuronen Layer C)
    multiplikator_C = (multiplikator_D @ dzD_daC) * daC_dzC # shape (4, 2) Anz_Samples, Anz_Neur_C
    d_theta_B = np.expand_dims(addBias(activations[-3]),1) * np.expand_dims(multiplikator_C,2) # shape: (4, 3, 2) Anz_Samples, Anz_Neur_D, Anz_Neur_C

    # Gradienten für ersten Layer (B)
    daB_dzB = activation_derivative(aB) # aB - np.square(aB) # (4, 2) - Anz_Samples, Anz_Neur_B
    dzC_daB = thetas[-2][:, 1:]#.T # (2, 2) - Anz_Neur_C, Anz_Neur_B    
    magic_matrix = (np.expand_dims(daC_dzC,1) * dzC_daB.T) @ thetas[-1][:, 1:].T    
    multiplikator_B = magic_matrix @ np.expand_dims(multiplikator_D,2) # shape (4, 2, 1)
    d_theta_X =  np.expand_dims(addBias(X), 1) * np.expand_dims(daB_dzB,2) * multiplikator_B # shape (4, 2, 3)

    return [d_theta_C, d_theta_B, d_theta_X]


def backpropagation_for_softmax(X, y, thetas, activations, linear_components, activation_derivative): 
    # Gradienten für Output-Layer
    dJ_do = activations[-1] - y
    multiplikator_D =  dJ_do # (4, 3) - Anz_Samples, Anz_Neur_D
    d_theta_C= np.expand_dims(addBias(activations[-2]), 1) * np.expand_dims(multiplikator_D, 2) # shape (4, 3, 3) Anz_Samples, Anz_Neur_

    # Gradienten für zweiten Layer (C)
    daC_dzC = activation_derivative(linear_components[-2]) # aC - np.square(aC) # (4, 2) - 4 samples, 2 Neuronen auf Layer C
    dzD_daC = thetas[-1][:, 1:] # Bias-Gewicht entfernen # (3, 2) - 3 Neuronen Layer D, mit je 2 thetas (Anz. Neuronen Layer C)
    multiplikator_C = (multiplikator_D @ dzD_daC) * daC_dzC # shape (4, 2) Anz_Samples, Anz_Neur_C
    d_theta_B = np.expand_dims(addBias(activations[-2]),1) * np.expand_dims(multiplikator_C,2) # shape: (4, 3, 2) Anz_Samples, Anz_Neur_D, Anz_Neur_C

    # Gradienten für ersten Layer (B)
    daB_dzB = activation_derivative(linear_components[-3]) # aB - np.square(aB) # (4, 2) - Anz_Samples, Anz_Neur_B
    dzC_daB = thetas[-2][:, 1:]#.T # (2, 2) - Anz_Neur_C, Anz_Neur_B    
    magic_matrix = (np.expand_dims(daC_dzC,1) * dzC_daB.T) @ thetas[-1][:, 1:].T    
    multiplikator_B = magic_matrix @ np.expand_dims(multiplikator_D,2) # shape (4, 2, 1)
    d_theta_X =  np.expand_dims(addBias(X), 1) * np.expand_dims(daB_dzB,2) * multiplikator_B # shape (4, 2, 3)

    return [d_theta_C, d_theta_B, d_theta_X]

def backpropagation_for_softmax_with_regularization(X, y, thetas, activations, linear_components, activation_derivative, _lambda):
    m = len(y) #Anzahl Samples
    multiplikator_reg = (_lambda/m )
    # Gradienten für Output-Layer
    dJ_do = activations[-1] - y
    multiplikator_D =  dJ_do # (4, 3) - Anz_Samples, Anz_Neur_D
    regularization_C = multiplikator_reg * thetas[-1]
    d_theta_C= np.expand_dims(addBias(activations[-2]), 1) * np.expand_dims(multiplikator_D, 2) + regularization_C # shape (4, 3, 3) Anz_Samples, Anz_Neur_

    # Gradienten für zweiten Layer (C)
    daC_dzC = activation_derivative(linear_components[-2]) # aC - np.square(aC) # (4, 2) - 4 samples, 2 Neuronen auf Layer C
    dzD_daC = thetas[-1][:, 1:] # Bias-Gewicht entfernen # (3, 2) - 3 Neuronen Layer D, mit je 2 thetas (Anz. Neuronen Layer C)
    multiplikator_C = (multiplikator_D @ dzD_daC) * daC_dzC # shape (4, 2) Anz_Samples, Anz_Neur_C
    regularization_B = multiplikator_reg * thetas[-2]
    d_theta_B = np.expand_dims(addBias(activations[-3]),1) * np.expand_dims(multiplikator_C,2) + regularization_B # shape: (4, 3, 2) Anz_Samples, Anz_Neur_D, Anz_Neur_C

    # Gradienten für ersten Layer (B)
    daB_dzB = activation_derivative(linear_components[-3]) # aB - np.square(aB) # (4, 2) - Anz_Samples, Anz_Neur_B
    dzC_daB = thetas[-2][:, 1:]#.T # (2, 2) - Anz_Neur_C, Anz_Neur_B    
    magic_matrix = (np.expand_dims(daC_dzC,1) * dzC_daB.T) @ thetas[-1][:, 1:].T    
    multiplikator_B = magic_matrix @ np.expand_dims(multiplikator_D,2) # shape (4, 2, 1)
    regularization_X = multiplikator_reg * thetas[-3]
    d_theta_X =  np.expand_dims(addBias(X), 1) * np.expand_dims(daB_dzB,2) * multiplikator_B + regularization_X # shape (4, 2, 3)

    return [d_theta_C, d_theta_B, d_theta_X]