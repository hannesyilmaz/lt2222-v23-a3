import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer


data = pd.read_csv('output.csv')
X = data.drop(["author", "test_train"], axis=1)
y = data["author"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def compute_average_precision_recall(y_true, y_pred):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binary = lb.transform(y_true)
    y_pred_binary = lb.transform(y_pred)
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(y_true_binary.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary[:, i], y_pred_binary[:, i])
        average_precision[i] = average_precision_score(y_true_binary[:, i], y_pred_binary[:, i])
    
    return precision, recall, average_precision


hidden_layer_sizes = [(50,), (100,), (200,), (300,)]
results = []

for size in hidden_layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=size, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    precision, recall, average_precision = compute_average_precision_recall(y_test, y_pred)
    results.append((size, precision, recall, average_precision))


plt.figure()
for size, precision, recall, average_precision in results:
    for i in range(len(precision)):
        plt.plot(recall[i], precision[i], label=f"Hidden Layer Size {size}, AP={average_precision[i]:.2f}")
        
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Different Hidden Layer Sizes")
plt.legend(loc="lower left")
plt.savefig("precision_recall_curve.png")
