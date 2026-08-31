import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import joblib
# Load all model responses
XS = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/largeSample/sciResponsesS.npy")
XM = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/largeSample/smResponsesS.npy")
XL = np.load('/Users/arahan/Desktop/Synopsys2025/npArrays/largeSample/litResponsesS.npy')

# Combine all model responses into feature matrix
X = np.column_stack((XS, XM, XL))
print(len(X))
# Load labels
y = np.load("/Users/arahan/Desktop/Synopsys2025/npArrays/largeSample/labelsS.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

corr_matrix = pd.DataFrame(X_train).corr()
print(corr_matrix)

# Train the Logistic Regression model
model = RandomForestClassifier(class_weight="balanced", n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Print evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "/Users/arahan/Desktop/Synopsys2025/CompilerModels/AcademicCompilerv3.pkl")