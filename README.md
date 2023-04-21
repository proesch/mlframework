# Machine Learning Framework
AutorInnen: Pascal Roesch, Natalie Vonier, Janis Roesser

## Willkommen!

Dieses Machine Learning Framework wurde von den AutorInnen im Wintersemester 2022/23 im Rahmen des Kurses "Machine Learning" am Institut für Mensch-Computer-Medien der Julius-Maximilians-Universität Würzburg entwickelt.  

Das Framework ermöglicht die Initiierung, das Training, die Evaluation und die Anwendung von neuronalen Netzen mit zwei verdeckten Schichten. Besonders hervorzuheben ist, dass das Framework außer dem Python Package "Pandas" (https://pandas.pydata.org/) keine externen Hilfen verwendet. Des Weiteren wurden die nötigen Berechnungen vektorisiert, was das Training und den Einsatz neuronaler Netze enorm beschleunigt.

Das Framework wurde von den AutorInnen verwendet, um ein neuronales Netz für die Erkennung von Gesten in einer live-Videoübertragung zu trainieren. Mit diesem neuronalen Netz konnte eine Diashow mittels Gesten gesteuert werden.

Das Präsentations-Video hierzu startet bei "Klick" auf dieses Bild:
[![Demo-Video Neu](gesture/Teaser_Image.jpg)](gesture/Machine_Learning_Framework_Roesch_Vonier_Roesser.mp4)


## Getting started

### Installieren benötigter Bibliotheken
```
pip install -r requirements.txt
```

Die Python-Skripte mnist_example.py und gesture_example.py geben einen verständlichen Einblick in die grundlegende Anwendung des Frameworks

### Daten und Parameter des Trainings definieren
```
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
```
### Struktur des neuronalen Netzes erstellen
Die Thetas werden automatisch mit Zufallswerten vorgefüllt:
```
trainer.createLayers([
    data_train.shape[1],        # Input-Neuronen
    20,                         # 1. Hidden Layer 
    20,                         # 2. Hidden Layer
    labels_train.shape[1]       # Output
])
```
### Hyperparameter wählen und Training starten:
```
trainer.trainNeuralNet(
    epochs=1000, 
    alpha=0.3
)
```
### Performance des Netzes auf dem Test-Datensatz ermitteln:
```
trainer.getTestDataPerformance(data_test, labels_test)
```

Darüber hinaus findet sich nachfolgend eine ausführliche Dokumentation der einzelnen Funktionalitäten und Parameter:

## Verfügbare Komponenten

### training.py
Dieses Skript ermöglicht die Instanziierung eines NeuralNetTrainer. Eine komfortable Möglichkeit, zum Training Neuronaler Netze mit nur wenigen Zeilen Code! Diese Klasse abstrahiert die unten folgenden Klassen und stellt einen einfachen Einstieg dar.

#### Konstruktor (Klasse NeuralNetTrainer)
| Parameter | Beschreibung | Basiert auf |
| --- | --- | --- |
| trainingData | Trainingsdaten | |
| trainingGroundTruth | Labels der Trainingsdaten | |
| validationData | Validierungsdaten| |
| validationGroundTruth | Labels der Validierungsdaten | |
| activationFunction | sigmoid, tanh oder relu | neuralnet.py |
| activationDerivative | sigmoid_derivative, tanh_derivative oder relu_derivative | neuralnet.py |
| forwardpassImplementation | forwardpass oder forwardpass_with_softmax | neuralnet.py |
| backpropImplementation | backpropagation_for_softmax oder backpropagation_for_softmax_with_regularization | neuralnet.py |
| gradientDescentImplementation | gradient_descent, gradient_descent_with_momentum oder gradient_descent_with_momentum_and_regularization | optimizers.py |
| lossFunction | categorical_cross_entropy, categorical_cross_entropy_with_regularization oder mse | metrics.py |

Die nachfolgenden optionalen Parameter können zur Verwendung zusätzlicher Optimierungsverfahren angegeben werden:
| Parameter | Beschreibung |
| --- | --- |
| _lambda | zur Kontrolle der Regularization, falls entsprechende Implementierungen ausgewählt wurden |
|beta | zur Kontrolle des Momentum |

#### Methoden
| Methode | Parameter | Beschreibung |
| --- | --- | --- |
| createLayers | n_neurons_per_layer | Akzeptiert eine Liste von Neuronen pro Schicht. Von der Eingabe- hin zur Ausgabeschicht, z.B. [784, 20, 20, 10]
| trainNeuralNet | epochs, alpha | Diese Methode liefert zudem den Verlauf des Loss, Accuracy und F1-Scores über den Trainingszeitraum hinweg in drei Rückgabeparametern zurück (unterteilt in Trainings- und Validierungsdaten je Epoche) |
| getTestDataPerformance | testData, testGroundTruth | Durchführung und Ausgabe der Evaluation des Testdatensatzes |

### data_helper.py
Ein einfaches Skript, welches Funktionen zur Datenverarbeitung bereitstellt:
| Methode | Parameter | Beschreibung |
| --- | --- | --- |
| create_one_hot | labels, nCategories | eindimensionale Ground Truth in One-Hot-Vektoren umwandeln |
| saveThetas | thetas, epoch, prefix="best" | Thetas im numpy-Format speichern (speichert eine Datei pro Schicht)
| loadThetas | layers | Liste mit Numpy-Dateien der Thetas, z.B. ['min_thetas_0.npy', 'min_thetas_1.npy', 'min_thetas_2.npy'] |

### feature_scaling.py
Stellt die Klasse StandardScaler mit folgenden Methoden zur Verfügung:
Ein einfaches Skript, welches Funktionen zur Datenverarbeitung bereitstellt:
| Methode | Parameter | Beschreibung |
| --- | --- | --- |
| fit | X | Den Scaler an die gegebenen Daten (X) anpassen |
| transform | X | Die Anpassung auf gegebene Stichproben anwenden |
| inverse_transform | X_scaled | Die Anwendung der Anpassung umkehren |

### metrics
Dieses Skript dient zwei grundlegenden Zwecken:
#### Bereitstellung der Kostenfunktionen:
| Methode | Parameter |
| --- | --- |
| cross_entropy | o, y_one_hot |
| categorical_cross_entropy | h_one_hot, y_one_hot |
| categorical_cross_entropy_with_regularization | h_one_hot, y_one_hot, _lambda, thetas |
| mse | h, y |
#### Bereitstellung der Evaluationsfunktionen
Diese führt NeuralNetTrainer bereits standardmäßig in jeder Epoche aus. Bei Bedarf können die Funktionen jedoch auch einzeln verwendet werden
| Methode | Parameter | Beschreibung |
| --- | --- | --- |
| classfiyHypothesis | h_vector | Bestimmt den Index des höchsten Ergebniswerts |
| classfiyHypothesisFiltered | h_one_hot, threshold | Bestimmt den Index des höchsten Ergebniswerts, falls dieser über einem bestimmtem Threshold ausfällt |
| getAccuracy | h_vector, y_one_hot | Bestimmt die Accuracy über alle Kategorien hinweg |
| f1_score | h, y | Bestimmt den f1-Score für zwei eindimensionale Vektoren |
| f1_scores_per_category | h_one_hot, y_one_hot | Bestimmt den f1-Score für jede Kategorie |
| f1_scores_mean | h_one_hot, y_one_hot | Bestimmt den über alle Kategorien gemittelten f1-Score |

### neuralnet.py
Die eigentliche Logik des Neuronalen Netzes. NeuralNetTrainer abstrahiert diese Funktionen. Sie können jedoch auch eigenständig verwendet werden.
| Methode | Parameter | Beschreibung |
| --- | --- | --- |
| generate_random_thetas | n_neurons, n_inputs_per_neuron, interval=1 | Für Intervall = 1 liegen die Zufallswerte zwischen -0.5 und 0.5 |
#### Aktivierungsfunktionen und zugehörige Ableitungen
| Methode | Parameter |
| --- | --- |
| sigmoid | z |
| sigmoid_derivative | z |
| tanh | z |
| tanh_derivative | z |
| relu | z |
| relu_derivative | z |
#### Forward Propagation
| Methode | Parameter |
| --- | --- |
| forwardpass | X, thetas, activation_function |
| forwardpass_with_softmax | X, thetas, activation_function |
#### Backpropagation
| Methode | Parameter |
| --- | --- |
| backpropagation | X, y, thetas, activations, linear_components, activation_derivative | 
| backpropagation_for_softmax | X, y, thetas, activations, linear_components, activation_derivative |
| backpropagation_for_softmax_with_regularization | X, y, thetas, activations, linear_components, activation_derivative, _lambda |

### optimizers.py
| Methode | Parameter |
| --- | --- |
| gradient_descent | gradients, thetas, alpha |
| gradient_descent_with_momentum | gradients, thetas, alpha, beta |
| gradient_descent_with_momentum_and_regularization | gradients, thetas, alpha, beta, _lambda |
