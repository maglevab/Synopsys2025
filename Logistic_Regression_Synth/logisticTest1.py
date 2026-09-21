import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

X = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/sciResponsesS.npy").reshape(-1, 1)
'''
XM = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/smResponsesS.npy")
XL = np.load('/Users/arahan/Desktop/Synopsys2025/npArrays/litResponsesS.npy')

X = np.column_stack((XS, XM, XL))
'''
y = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/labelsS.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test.ravel(), y=y_pred.ravel(), hue=y_test, palette={0: 'blue', 1: 'red'}, marker='o')
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

print("Shape of X_test:", X_test.shape)
print("\nFirst few rows of X_test:")
print(X_test[:5])

plt.show()
