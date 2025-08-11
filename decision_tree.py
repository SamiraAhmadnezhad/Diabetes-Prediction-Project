# =================================================================
#               FINAL SCRIPT 1: OPTIMIZED DECISION TREE
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

print("--- Script for Optimized Decision Tree Started ---")

# Step 2: Load and Prepare Data
print("\n[Step 2] Loading and Preparing Data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)

# Replace 0 with NaN where it's medically impossible
cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)
print("Data loaded and initial cleaning done.")
print(f"Dataset shape: {df.shape}")
print(f"Missing values after cleaning:\n{df.isnull().sum()}")

# Step 3: Split Data and Impute Missing Values Correctly
print("\n[Step 3] Splitting data and imputing missing values...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data BEFORE any imputation to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Calculate median from the training set ONLY
median_values = X_train.median()
print(f"\nMedian values calculated from training data:\n{median_values}")

# Impute missing values in both sets using the calculated median
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
print("Missing values imputed successfully.")

# Step 4: Hyperparameter Tuning with GridSearchCV
print("\n[Step 4] Finding the best Decision Tree model with GridSearchCV...")
# Expanded parameter grid for a more thorough search
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [10, 15, 20, 25],
    'min_samples_split': [20, 30, 40]
}

dt_model = DecisionTreeClassifier(random_state=42)

# Using a weighted F1-score to handle imbalance
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)

grid_search_dt.fit(X_train, y_train)

# Get the best model
best_dt_model = grid_search_dt.best_estimator_
print("\nBest parameters found for Decision Tree:")
print(grid_search_dt.best_params_)

# Step 5: Dual Evaluation (Train vs. Test)
print("\n[Step 5] Performing dual evaluation to check for overfitting...")

# 5.1: Evaluation on Test Data
print("\n--- Performance on Test Set (Unseen Data) ---")
y_pred_test_dt = best_dt_model.predict(X_test)
print(classification_report(y_test, y_pred_test_dt, target_names=['Class 0 (Healthy)', 'Class 1 (Diabetic)']))
accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
f1_test_dt = classification_report(y_test, y_pred_test_dt, output_dict=True)['weighted avg']['f1-score']

# 5.2: Evaluation on Train Data
print("\n--- Performance on Train Set (Seen Data) ---")
y_pred_train_dt = best_dt_model.predict(X_train)
print(classification_report(y_train, y_pred_train_dt, target_names=['Class 0 (Healthy)', 'Class 1 (Diabetic)']))
accuracy_train_dt = accuracy_score(y_train, y_pred_train_dt)
f1_train_dt = classification_report(y_train, y_pred_train_dt, output_dict=True)['weighted avg']['f1-score']

# 5.3: Final Comparison and Conclusion
print("\n--- Final Overfitting Analysis for Decision Tree ---")
print(f"Test Accuracy: {accuracy_test_dt:.2%}")
print(f"Train Accuracy: {accuracy_train_dt:.2%}")
print(f"Test F1-Score: {f1_test_dt:.2%}")
print(f"Train F1-Score: {f1_train_dt:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train_dt - f1_test_dt):.2%}")
print("\n--- End of Decision Tree Script ---")