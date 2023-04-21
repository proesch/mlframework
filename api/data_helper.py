import numpy as np

def create_one_hot(labels, nCategories):
    out = np.zeros((labels.size, nCategories))
    for i, label in enumerate(labels):
        out[i][label] = 1
    return out

# Gegebene Thetas in Datei (je Layer) speichern
def saveThetas(thetas, epoch, prefix="best"):
    for index, theta in enumerate(thetas):
        np.save(prefix + "_thetas_" + str(index), theta)
    print("Best thetas yet saved for epoch ", epoch)

# Input: Liste mit Numpy-Dateien der Thetas, z.B. ['min_thetas_0.npy', 'min_thetas_1.npy', 'min_thetas_2.npy']
def loadThetas(layers):
    thetas = []
    for layer in layers:
        thetas.append(np.load(layer))
    return thetas