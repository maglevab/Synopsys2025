import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
# Load all model responses
XS = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/sciResponsesM.npy")
XM = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/smResponsesM.npy")
XL = np.load('/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/litResponsesM.npy')

# Combine all model responses into feature matrix
X = np.column_stack((XS, XM, XL))
XS = XS.astype(int)
XM = XM.astype(int)
XL = XL.astype(int)
print(np.bincount(XS), np.bincount(XM), np.bincount(XL))
print(len(X))
# Load labels
y = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/NewExpert/labelsM.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


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

joblib.dump(model, "/Users/arahan/Desktop/Synopsys2025/CompilerModelsv2/MediaCompiler.pkl")
importances = mutual_info_classif(X, y)
print(importances)
'''
# Compute predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression Model")
plt.legend()
plt.grid()

corr_matrix = pd.DataFrame(X_train).corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
'''