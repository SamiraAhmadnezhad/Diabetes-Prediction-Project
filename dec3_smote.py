# =================================================================
#       FINAL SCRIPT 6: DECISION TREE WITH CONTROLLED SMOTE
# Goal: Improve Recall by generating synthetic data for the minority class.
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score
from collections import Counter

print("--- Script for Decision Tree with Controlled SMOTE Started ---")

# Step 2 & 3: Load, Clean, and Engineer Features
# (These steps are identical to the previous script)
print("\n[Step 2 & 3] Loading, Cleaning, and Engineering Features...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)
cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)
df['Glucose_Insulin_Ratio'] = df['glucose'] / df['insulin']
df['BMI_Age'] = df['mass'] * df['age']
df['Pregnancy_Age'] = df['pregnant'] * df['age']
df['BMI_Category'] = df['mass'].apply(lambda bmi: 'Underweight' if bmi < 18.5 else ('Normal' if 18.5 <= bmi < 25 else ('Overweight' if 25 <= bmi < 30 else 'Obese')))
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
print("Data pipeline prep completed.")

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
print("Data scaled.")

# Step 5: Controlled SMOTE
print("\n[Step 5] Applying controlled SMOTE...")
# We bring the minority class to 80% of the majority, not 100%
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Class distribution after SMOTE: {sorted(Counter(y_train_resampled).items())}")

# Step 6: Hyperparameter Tuning on Resampled Data
print("\n[Step 6] Finding the best Decision Tree on SMOTE data...")
param_dist_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(4, 15)),
    'min_samples_leaf': list(range(10, 31)),
    'min_samples_split': list(range(20, 51))
}
dt_model = DecisionTreeClassifier(random_state=42)
random_search_dt = RandomizedSearchCV(estimator=dt_model, param_distributions=param_dist_dt, n_iter=100,
                                      cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42)
random_search_dt.fit(X_train_resampled, y_train_resampled)
best_dt_smote_model = random_search_dt.best_estimator_
print("\nBest parameters found:")
print(random_search_dt.best_params_)

# Step 7: Dual Evaluation
print("\n[Step 7] Performing dual evaluation...")
y_pred_test = best_dt_smote_model.predict(X_test_scaled)
y_pred_train = best_dt_smote_model.predict(X_train_scaled)
f1_test = f1_score(y_test, y_pred_test, average='weighted')
f1_train = f1_score(y_train, y_pred_train, average='weighted')

print("\n--- Performance on Test Set ---")
print(classification_report(y_test, y_pred_test))
print("\n--- Final Overfitting Analysis ---")
print(f"Test F1-Score: {f1_test:.2%}")
print(f"Train F1-Score: {f1_train:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train - f1_test):.2%}")