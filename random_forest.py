# =================================================================
#   FINAL SCRIPT 4: RANDOM FOREST WITH FEATURE ENGINEERING & SMOTE
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

print("--- Script for Advanced Random Forest Started ---")

# Step 2: Load and Prepare Data
print("\n[Step 2] Loading and Preparing Data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)

cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)
print("Data loaded and initial cleaning done.")

# Step 3: Feature Engineering
print("\n[Step 3] Performing Feature Engineering...")
df['Glucose_Insulin_Ratio'] = df['glucose'] / df['insulin']
df['BMI_Age'] = df['mass'] * df['age']
df['Pregnancy_Age'] = df['pregnant'] * df['age']
df['BMI_Category'] = df['mass'].apply(lambda bmi: 'Underweight' if bmi < 18.5 else ('Normal' if 18.5 <= bmi < 25 else ('Overweight' if 25 <= bmi < 30 else 'Obese')))
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
print("New features created.")

# Step 4: Split, Impute, and Scale Data
print("\n[Step 4] Splitting, imputing, and scaling data...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
median_values = X_train.median()
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data pipeline completed.")

# Step 5: Handle Class Imbalance with SMOTE
print("\n[Step 5] Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Class distribution after SMOTE: {sorted(Counter(y_train_resampled).items())}")

# Step 6: Hyperparameter Tuning with RandomizedSearchCV for Random Forest
print("\n[Step 6] Finding the best Random Forest model with RandomizedSearchCV...")
# Very wide parameter space
param_dist_rf = {
    'n_estimators': list(range(100, 401, 50)),
    'max_depth': list(range(8, 21)),
    'min_samples_leaf': list(range(2, 8)),
    'min_samples_split': list(range(5, 16)),
    'bootstrap': [True, False], # Test with and without bootstrap
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=42)
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist_rf, n_iter=100,
                                      cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42)
random_search_rf.fit(X_train_resampled, y_train_resampled)
best_rf_model = random_search_rf.best_estimator_
print("\nBest parameters found for Random Forest:")
print(random_search_rf.best_params_)

# Step 7: Dual Evaluation
print("\n[Step 7] Performing dual evaluation...")
y_pred_test_rf = best_rf_model.predict(X_test_scaled)
y_pred_train_rf = best_rf_model.predict(X_train_scaled)
f1_test_rf = classification_report(y_test, y_pred_test_rf, output_dict=True)['weighted avg']['f1-score']
f1_train_rf = classification_report(y_train, y_pred_train_rf, output_dict=True)['weighted avg']['f1-score']

print("\n--- Performance on Test Set (Random Forest) ---")
print(classification_report(y_test, y_pred_test_rf))
print("\n--- Final Overfitting Analysis (Random Forest) ---")
print(f"Test F1-Score: {f1_test_rf:.2%}")
print(f"Train F1-Score: {f1_train_rf:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train_rf - f1_test_rf):.2%}")