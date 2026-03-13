import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preprocessing

# Load dataset from CSV file into a pandas DataFrame
df = pd.read_csv("Loan_Default.csv")

# Remove duplicate rows to avoid bias in model training
df = df.drop_duplicates()

# Handle missing values in numeric columns by filling them with the median
# This ensures no empty nans remain which would cause errors during model training
df = df.fillna(df.median(numeric_only=True))

# Identify categorical columns (object or string types)
# These columns will be converted to numeric labels
cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()

# Initialize LabelEncoder to convert categorical features to numeric
le = LabelEncoder()

# Apply label encoding to each categorical column
# This is necessary because scikit-learn models require numeric input
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Step 2: Feature and Label Definition


# Define target variable 'y' as the Status column
# 1 = High Risk (likely default), 0 = Low Risk (likely repay)
y = df['Status']

# Define feature matrix 'X' by dropping the target column
# All remaining columns are treated as predictors
X = df.drop('Status', axis=1)

# Step 3: Train-Test Split

# Split the dataset into training (80%) and testing (20%) sets
# stratify=y ensures the class distribution is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numerical features to have mean=0 and std=1
# Fit on training data and transform both training and test sets
# Scaling improves convergence and performance of models like Logistic Regression and SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Model Training

# Initialize Logistic Regression model
# solver='lbfgs' handles large datasets efficiently, max_iter increased for convergence
log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Train Logistic Regression on training data
log_model.fit(X_train, y_train)

# Initialize Support Vector Machine model with a linear kernel
# Linear kernel is faster for large datasets and often sufficient for classification
svm_model = SVC(kernel='linear')

# Train SVM on training data
svm_model.fit(X_train, y_train)

# Step 5: Evaluation

# Make predictions on the test set using both models
y_pred_log = log_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)

# Calculate and print accuracy for both models
# Accuracy = (correct predictions) / (total predictions)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Generate confusion matrices to evaluate classification performance
# Shows True Negatives, False Positives, False Negatives, True Positives
cm_log = confusion_matrix(y_test, y_pred_log)
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Plot confusion matrices side by side using seaborn heatmaps
# Provides a visual representation of model performance
fig, axes = plt.subplots(1, 2, figsize=(12,5))


# Logistic Regression confusion matrix
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# SVM confusion matrix
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("SVM Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Adjust layout to prevent overlapping labels and show plots
plt.tight_layout()
plt.show()


