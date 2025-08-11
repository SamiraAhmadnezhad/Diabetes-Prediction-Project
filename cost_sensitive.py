# =================================================================
# FINAL SCRIPT 5: COST-SENSITIVE DECISION TREE
# Goal: Improve Recall by penalizing misclassification of the minority class.
# =================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score

print("--- Script for Cost-Sensitive Decision Tree Started ---")

# Step 2: Load and Prepare Data
print("\n[Step 2] Loading and Preparing Data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)
cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)
print("Data loaded and cleaned.")

# Step 3: Feature Engineering
print("\n[Step 3] Performing Feature Engineering...")
df['Glucose_Insulin_Ratio'] = df['glucose'] / df['insulin']
df['BMI_Age'] = df['mass'] * df['age']
df['Pregnancy_Age'] = df['pregnant'] * df['age']
df['BMI_Category'] = df['mass'].apply(lambda bmi: 'Underweight' if bmi < 18.5 else ('Normal' if 18.5 <= bmi < 25 else ('Overweight' if 25 <= bmi < 30 else 'Obese')))
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
print("New features created.")

# Step 4: Split Data and Impute Missing Values
print("\n[Step 4] Splitting data and imputing missing values...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
median_values = X_train.median()
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
print("Data pipeline completed.")

# Step 5: Hyperparameter Tuning with Cost-Sensitive Learning
print("\n[Step 5] Finding the best Cost-Sensitive Decision Tree with RandomizedSearchCV...")
param_dist_dt_cs = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(5, 15)),
    'min_samples_leaf': list(range(10, 31)),
    'min_samples_split': list(range(20, 51)),
    # This is the key: we give more weight to the minority class (1)
    'class_weight': [{0: 1, 1: w} for w in [1.5, 2, 2.5, 3, 3.5]]
}

dt_model = DecisionTreeClassifier(random_state=42)
random_search_dt_cs = RandomizedSearchCV(estimator=dt_model, param_distributions=param_dist_dt_cs, n_iter=100,
                                         cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42)
random_search_dt_cs.fit(X_train, y_train)
best_dt_cs_model = random_search_dt_cs.best_estimator_
print("\nBest parameters found for Cost-Sensitive Decision Tree:")
print(random_search_dt_cs.best_params_)

# Step 6: Dual Evaluation
print("\n[Step 6] Performing dual evaluation...")
y_pred_test = best_dt_cs_model.predict(X_test)
y_pred_train = best_dt_cs_model.predict(X_train)
f1_test = f1_score(y_test, y_pred_test, average='weighted')
f1_train = f1_score(y_train, y_pred_train, average='weighted')

print("\n--- Performance on Test Set ---")
print(classification_report(y_test, y_pred_test))
print("\n--- Final Overfitting Analysis ---")
print(f"Test F1-Score: {f1_test:.2%}")
print(f"Train F1-Score: {f1_train:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train - f1_test):.2%}")