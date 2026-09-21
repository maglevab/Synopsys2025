import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load all model responses
XS = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/sciResponsesS.npy")
XM = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/smResponsesS.npy")
XL = np.load('/Users/arahan/Desktop/Synopsys2025/npArrays/litResponsesS.npy')

# Combine all model responses into feature matrix
X = np.column_stack((XS, XM, XL))

# Load labels
y = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/labelsS.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# No need for reshape since we have multiple features
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Create visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette={
    0: 'blue', 1: 'red'}, marker='o', alpha=0.5, s=50)
plt.xlabel("Scientific Model Score")
plt.ylabel("Media Model Score")
plt.title("AI Text Classification Decision Boundary\nAccuracy: {:.2f}%".format(
    accuracy * 100))
plt.legend(title="Human/AI", loc="upper right")

# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")

# Print data shapes for verification
print("Shape of X_test:", X_test.shape)
print("\nFirst few rows of X_test:")
print(X_test[:5])


def plot_decision_boundary(X, y, model):
    # Create mesh grid for smooth curve
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Get predictions for all points in mesh grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create contour plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
    plt.colorbar(label='Probability of being AI-generated')

    # Plot original data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette={
        0: 'blue', 1: 'red'}, marker='o', alpha=0.5, s=50)

    plt.xlabel("Scientific Model Score")
    plt.ylabel("Media Model Score")
    plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(
        accuracy * 100))
    plt.legend(title="Human/AI", loc="upper right")


# Create visualization
plt.figure(figsize=(8, 6))
plot_decision_boundary(X_test, y_test, model)
plt.show()

plt.show()