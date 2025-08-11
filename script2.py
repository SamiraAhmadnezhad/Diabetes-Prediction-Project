# =================================================================
#   FINAL ANALYSIS: TESTING ADVANCED TECHNIQUES WITH CORRECT METHODOLOGY
#   This is the "Correct" script (Code #3)
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

# Suppress future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

print("--- Analysis Script with Correct Methodology Started ---")

# Step 2: Load and Prepare Initial Data
print("\n[Step 2] Loading and preparing initial data...")
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
df.loc[:, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    df.loc[:, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print("Data loaded and initial cleaning done.")

# Step 3: Initial Data Split (THE MOST IMPORTANT STEP)
print("\n[Step 3] Performing initial Train-Test Split to prevent data leakage...")
X = df.drop('Outcome', axis=1)
y = df['Outcome']
# Using a 70/30 split to be closer to the other scripts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Initial Train shape: {X_train.shape}, Initial Test shape: {X_test.shape}")

# Step 4: Preprocessing applied ONLY to the Training Set
print("\n[Step 4] Applying advanced preprocessing ONLY on the training set...")

# 4.1: Grouped Imputation on Training Data
# We create a temporary training dataframe to perform the grouped imputation
train_df = pd.concat([X_train, y_train], axis=1)
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Calculate means from the training data groups
# This is tricky. We need to handle the case where a group might not have any non-NaN values.
# A more robust way is to calculate the means first.
grouped_means = train_df.groupby('Outcome')[columns_to_impute].mean()

# Fill NaNs in X_train based on the group
for col in columns_to_impute:
    X_train[col] = X_train[col].fillna(y_train.map(grouped_means[col]))

# If any NaNs still exist (e.g., a group had no valid data), fill with overall train median
X_train.fillna(X_train.median(), inplace=True)
print("Grouped imputation completed on training data.")

# 4.2: Outlier Removal on Training Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform the training data

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_train_scaled)
distances, _ = knn.kneighbors(X_train_scaled)
outlier_scores = distances.mean(axis=1)
threshold = np.percentile(outlier_scores, 95)

# Create a mask to remove outliers ONLY from the training set
mask = outlier_scores < threshold
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]
print(f"Training data size before outlier removal: {len(X_train)}. After removal: {len(X_train_clean)}")

# Step 5: Preprocessing the Test Set using knowledge from Training Set
print("\n[Step 5] Preparing the test set using insights from the training set...")
# We cannot use grouped imputation on the test set.
# We must use a single, robust value calculated from the training set.
test_imputation_values = X_train.median() # Use median from the original imputed train set
X_test.fillna(test_imputation_values, inplace=True)
print("Test set imputed with training data median.")
# We do NOT remove outliers from the test set.

# Step 6: Hyperparameter Tuning and Training
print("\n[Step 6] Finding the best model and training on the cleaned training data...")
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# Train on the cleaned training data
grid_search.fit(X_train_clean, y_train_clean)
best_model = grid_search.best_estimator_
print("\nBest parameters found:")
print(grid_search.best_params_)

# Step 7: Final Evaluation on UNTOUCHED Test Set
print("\n[Step 7] Evaluating the final model on the untouched test set...")
# We must scale the test data before prediction
X_test_scaled = scaler.transform(X_test) # Note: we only transform, not fit_transform
y_pred_test = best_model.predict(X_test)
print("\n--- Performance on Test Set ---")
print(classification_report(y_test, y_pred_test, target_names=['Class 0 (Healthy)', 'Class 1 (Diabetic)']))
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='weighted')
print(f"Final Test Accuracy: {accuracy_test:.2%}")
print(f"Final Test F1-Score: {f1_test:.2%}")

# Optional: Check performance on the cleaned train set to estimate overfitting
print("\n--- Performance on the Cleaned Train Set (for Overfitting check) ---")
y_pred_train_clean = best_model.predict(X_train_clean)
f1_train_clean = f1_score(y_train_clean, y_pred_train_clean, average='weighted')
print(f"Cleaned Train F1-Score: {f1_train_clean:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train_clean - f1_test):.2%}")

print("\n--- Analysis with Correct Methodology Finished ---")