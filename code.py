import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from joblib import dump

# Load dataset from the generated CSV file
dataset = pd.read_csv('dataset.csv')

# Extract features (N, P, K, temperature, humidity) and labels (label)
X = dataset[['N', 'P', 'K', 'temperature', 'humidity']].values
y = dataset['label'].values

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_classifier.fit(X_train, y_train)

# Predict labels for the training set
y_train_pred = mlp_classifier.predict(X_train)

# Predict labels for the test set
y_test_pred = mlp_classifier.predict(X_test)

# Compute accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Compute loss during training
train_loss = mlp_classifier.loss_curve_

# Plot confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot training loss curve
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and testing accuracy
plt.figure(figsize=(8, 4))
plt.plot([train_accuracy] * len(train_loss), label='Training Accuracy')
plt.plot([test_accuracy] * len(train_loss), label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model to a .joblib file
dump(mlp_classifier, 'mlp_classifier.joblib')
