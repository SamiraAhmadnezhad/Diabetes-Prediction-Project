# =================================================================
#       FINAL SCRIPT 3: DECISION TREE WITH FEATURE ENGINEERING
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

print("--- Script for Decision Tree with Feature Engineering Started ---")

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

# Create new interaction features (handle potential division by zero)
df['Glucose_Insulin_Ratio'] = df['glucose'] / df['insulin']
df['BMI_Age'] = df['mass'] * df['age']
df['Pregnancy_Age'] = df['pregnant'] * df['age']

# Create categorical features
def categorize_bmi(bmi):
    if bmi < 18.5: return 'Underweight'
    if 18.5 <= bmi < 25: return 'Normal'
    if 25 <= bmi < 30: return 'Overweight'
    return 'Obese'
df['BMI_Category'] = df['mass'].apply(categorize_bmi)

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
print("New features created.")
print(f"New dataset shape: {df.shape}")

# Step 4: Split Data and Impute Missing Values Correctly
print("\n[Step 4] Splitting data and imputing missing values...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Impute based on training data median
median_values = X_train.median()
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
print("Data split and imputed.")

# Step 5: Hyperparameter Tuning with RandomizedSearchCV
print("\n[Step 5] Finding the best Decision Tree model with RandomizedSearchCV...")
# Wider parameter space
param_dist_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(4, 15)),
    'min_samples_leaf': list(range(10, 31)),
    'min_samples_split': list(range(20, 51))
}

dt_model = DecisionTreeClassifier(random_state=42)

# Use RandomizedSearchCV with 100 iterations
random_search_dt = RandomizedSearchCV(estimator=dt_model, param_distributions=param_dist_dt, n_iter=100,
                                      cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42)
random_search_dt.fit(X_train, y_train)

best_dt_model = random_search_dt.best_estimator_
print("\nBest parameters found for Decision Tree:")
print(random_search_dt.best_params_)

# Step 6: Dual Evaluation
print("\n[Step 6] Performing dual evaluation...")
# (The rest of the evaluation code is the same as before)
y_pred_test_dt = best_dt_model.predict(X_test)
y_pred_train_dt = best_dt_model.predict(X_train)
f1_test_dt = classification_report(y_test, y_pred_test_dt, output_dict=True)['weighted avg']['f1-score']
f1_train_dt = classification_report(y_train, y_pred_train_dt, output_dict=True)['weighted avg']['f1-score']

print("\n--- Performance on Test Set (Decision Tree) ---")
print(classification_report(y_test, y_pred_test_dt))
print("\n--- Final Overfitting Analysis (Decision Tree) ---")
print(f"Test F1-Score: {f1_test_dt:.2%}")
print(f"Train F1-Score: {f1_train_dt:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train_dt - f1_test_dt):.2%}")