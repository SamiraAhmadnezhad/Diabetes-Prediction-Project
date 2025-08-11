# =================================================================
#           FINAL SCRIPT 2: OPTIMIZED RANDOM FOREST + SMOTE
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

print("--- Script for Optimized Random Forest with SMOTE Started ---")

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

# Step 3: Split, Impute, and Scale Data Correctly
print("\n[Step 3] Splitting, imputing, and scaling data...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split data BEFORE any other processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Impute missing values based on training set median
median_values = X_train.median()
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
print("Missing values imputed successfully.")

# Scale data based on training set statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled successfully.")

# Step 4: Handle Class Imbalance with SMOTE
print("\n[Step 4] Handling class imbalance with SMOTE...")
print(f"Class distribution before SMOTE: {sorted(Counter(y_train).items())}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Class distribution after SMOTE: {sorted(Counter(y_train_resampled).items())}")

# Step 5: Hyperparameter Tuning with GridSearchCV for Random Forest
print("\n[Step 5] Finding the best Random Forest model with GridSearchCV...")
# Expanded parameter grid
param_grid_rf = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [8, 10, 12, 14],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12]
}

rf_model = RandomForestClassifier(random_state=42)

grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)

# Fit on the resampled (balanced and scaled) data
grid_search_rf.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_rf_model = grid_search_rf.best_estimator_
print("\nBest parameters found for Random Forest:")
print(grid_search_rf.best_params_)

# Step 6: Dual Evaluation (Train vs. Test)
print("\n[Step 6] Performing dual evaluation to check for overfitting...")

# 6.1: Evaluation on Test Data (Unseen and Unscaled)
print("\n--- Performance on Test Set (Unseen Data) ---")
y_pred_test_rf = best_rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_test_rf, target_names=['Class 0 (Healthy)', 'Class 1 (Diabetic)']))
accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
f1_test_rf = classification_report(y_test, y_pred_test_rf, output_dict=True)['weighted avg']['f1-score']

# 6.2: Evaluation on Original Train Data (to get a realistic overfitting measure)
print("\n--- Performance on Train Set (Original Unbalanced Data) ---")
# We predict on the original scaled training data
y_pred_train_rf = best_rf_model.predict(X_train_scaled)
print(classification_report(y_train, y_pred_train_rf, target_names=['Class 0 (Healthy)', 'Class 1 (Diabetic)']))
accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
f1_train_rf = classification_report(y_train, y_pred_train_rf, output_dict=True)['weighted avg']['f1-score']

# 6.3: Final Comparison and Conclusion
print("\n--- Final Overfitting Analysis for Random Forest ---")
print(f"Test Accuracy: {accuracy_test_rf:.2%}")
print(f"Train Accuracy: {accuracy_train_rf:.2%}")
print(f"Test F1-Score: {f1_test_rf:.2%}")
print(f"Train F1-Score: {f1_train_rf:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train_rf - f1_test_rf):.2%}")
print("\n--- End of Random Forest Script ---")