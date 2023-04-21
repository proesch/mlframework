import numpy as np
# ----------------
# Gradient Descent
# ----------------
def gradient_descent(gradients, thetas, alpha):
    #Mittelwerte der Gradienten berechnen       
    for index, layer in enumerate(gradients):
        layer = np.mean(layer, axis=0)
        thetas[len(thetas)-1-index] -= alpha * layer # Thetas werden direkt manipuliert
    return thetas

def gradient_descent_with_momentum(gradients, thetas, alpha, beta):
    #Mittelwerte der Gradienten berechnen       
    for index, layer in enumerate(gradients):
        layer = np.mean(layer, axis=0)
        velocity = alpha * layer
        velocity += beta * velocity
        thetas[len(thetas)-1-index] -= velocity 
    return thetas

def gradient_descent_with_momentum_and_regularization(gradients, thetas, alpha, beta, _lambda):
    #Mittelwerte der Gradienten berechnen
    for index, layer in enumerate(gradients):
        layer = np.mean(layer, axis=0)
        velocity = alpha * (layer + (_lambda/len(gradients[0]) * thetas[len(thetas)-1-index])) # richtige Länge wäre anzahl samples
        velocity += beta * velocity
        thetas[len(thetas)-1-index] -= velocity 
    return thetas