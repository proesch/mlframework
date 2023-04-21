import numpy as np

# ---------------
# Error-Functions
# ---------------
# Gut geeignet für Eindimensionalen-Output, z.B. binäre Klassifikation
def cross_entropy(o, y_one_hot):
    J = np.mean(- y_one_hot * np.log(o.flatten()) - (1 - y_one_hot) * np.log(1 - o.flatten()))
    return J

# Sowohl h_one_hot als auch y_one_hot sollten die gleiche, mehrdimensionale Shape aufweisen (1 Output-Neuron je Kategorie)
def categorical_cross_entropy(h_one_hot, y_one_hot):
    h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)
    m = len(h_one_hot)
    return -1/m * np.sum(y_one_hot * np.log(h_one_hot))

def categorical_cross_entropy_with_regularization(h_one_hot, y_one_hot, _lambda, thetas):
    h_one_hot = np.clip(h_one_hot, a_min=0.000000001, a_max=None)
    m = len(h_one_hot)
    cost_regularization = _lambda/(2 * m) * (np.sum(np.square(thetas[0])) + np.sum(np.square(thetas[1])) + np.sum(np.square(thetas[2])))
    return (-1/m * np.sum(y_one_hot * np.log(h_one_hot))) + cost_regularization

def mse(h, y):
    return np.mean((h - y)**2)

# ----------
# Evaluation
# ----------

def classfiyHypothesis(h_one_hot):
    return np.argmax(h_one_hot, axis=1)

def classfiyHypothesisFiltered(h_one_hot, threshold):
    highest_values_index = np.argmax(h_one_hot, axis=1)
    highest_values = np.max(h_one_hot, axis=1)
    result = np.zeros_like(highest_values)

    for i in range(h_one_hot.shape[0]):
        
        if highest_values[i] < threshold:
            result[i] = 0
        else:
            result[i] = highest_values_index[i]

    return result

def getAccuracy(h_vector, y_one_hot):
    h_int = classfiyHypothesis(h_vector)
    y_int = classfiyHypothesis(y_one_hot)
    acc  = (h_int == y_int).mean()
    return acc

def generateMetricsHistory(n_epochs):
    error_history = np.zeros((n_epochs, 2)) # [0] Training_error, [1] Validation_error
    accuracy_history = np.zeros((n_epochs, 2)) # [0] Training_accuracy, [1] Validation_accuracy
    return error_history, accuracy_history

# F1 Score
def f1_score(h_one_hot, y_one_hot): # beide Vektoren müssen als one_hot_vector übergeben werden
    true_positives = (h_one_hot == 1) & (y_one_hot == 1)
    false_positives = (h_one_hot == 1) & (y_one_hot == 0)
    false_negatives = (h_one_hot == 0) & (y_one_hot == 1)

    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    
    f1 = 2 * (precision * recall/(precision + recall))
    
    return f1

def f1_scores_per_category(h_one_hot, y_one_hot):
    n_categories = y_one_hot.shape[1]
    f1_scores = np.zeros(n_categories)
    for category in range(n_categories):
        h = classfiyHypothesis(h_one_hot) == category
        y = classfiyHypothesis(y_one_hot) == category
        f1_scores[category] = f1_score(h, y)
    return f1_scores

def f1_scores_mean(h, y):
    return (np.asarray(f1_scores_per_category(h, y))).mean()