# =================================================================
#   FINAL SCRIPT 6: REGULARIZED RANDOM FOREST TO REDUCE OVERFITTING
# =================================================================

# Step 1-3: Import Libraries, Load Data, Feature Engineering (Same as before)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel # ✅ Import feature selector

print("--- Script for Regularized Random Forest Started ---")

# ... (کد مربوط به بارگذاری داده و مهندسی ویژگی بدون تغییر در اینجا قرار می‌گیرد) ...
# Step 2: Load and Prepare Data
print("\n[Step 2] Loading and Preparing Data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)
df.loc[:, ['glucose', 'pressure', 'triceps', 'insulin', 'mass']] = df.loc[:, ['glucose', 'pressure', 'triceps', 'insulin', 'mass']].replace(0, np.nan)
print("Data loaded and initial cleaning done.")

# Step 3: Feature Engineering
print("\n[Step 3] Performing Feature Engineering...")
df['Glucose_Insulin_Ratio'] = df['glucose'] / df['insulin']
df['BMI_Age'] = df['mass'] * df['age']
df['Pregnancy_Age'] = df['pregnant'] * df['age']

def categorize_bmi(bmi):
    if pd.isnull(bmi): return np.nan
    if bmi < 18.5: return 'Underweight'
    if 18.5 <= bmi < 25: return 'Normal'
    if 25 <= bmi < 30: return 'Overweight'
    return 'Obese'
df['BMI_Category'] = df['mass'].apply(categorize_bmi)
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
print(f"New dataset shape: {df.shape}")


# Step 4: Split Data
print("\n[Step 4] Splitting data...")
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Step 5: Build an Advanced Pipeline with Feature Selection
print("\n[Step 5] Building Pipeline with Feature Selection and Regularization...")

# We add SelectFromModel to the pipeline
pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('selector', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced_subsample')) # 'balanced_subsample' is often better
])

# ✅ New, more constrained parameter grid to encourage simpler models
param_dist_rf_regularized = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [4, 5, 6, 7, 8 , 10],            
    'rf__min_samples_leaf': [5, 10, 15, 20], #  elemenet 5          
    'rf__min_samples_split': [10, 20, 30, 40],        
    'rf__max_features': ['sqrt', 'log2', 0.5],      
    'selector__threshold': ['median', '1.5*mean'] ######################################I deleted 'mean'
}

# Perform Randomized Search with the new parameters
random_search_rf = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist_rf_regularized,
    n_iter=100,
    cv=8,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search_rf.fit(X_train, y_train)

best_model = random_search_rf.best_estimator_
print("\nBest regularized parameters found for RandomForest Pipeline:")
print(random_search_rf.best_params_)

# Step 6 & 7: Threshold Tuning and Final Evaluation (Same as before)
print("\n[Step 6] Finding the optimal threshold...")

y_proba_test = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)

f1_scores = (2 * precisions * recalls) / (precisions + recalls)
f1_scores = f1_scores[:-1]
thresholds = thresholds[:len(f1_scores)]

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold found: {optimal_threshold:.4f} (Maximizes F1-Score)")

y_pred_optimal = (y_proba_test >= optimal_threshold).astype(int)

print("\n--- Performance with Optimal Threshold on Test Set ---")
print(classification_report(y_test, y_pred_optimal))

# Overfitting Analysis
y_proba_train = best_model.predict_proba(X_train)[:, 1]
y_pred_train_optimal = (y_proba_train >= optimal_threshold).astype(int)
f1_test = f1_score(y_test, y_pred_optimal, average='weighted')
f1_train = f1_score(y_train, y_pred_train_optimal, average='weighted')

print("\n--- Final Overfitting Analysis (Regularized Random Forest) ---")
print(f"Test F1-Score: {f1_test:.2%}")
print(f"Train F1-Score: {f1_train:.2%}")
print(f"Performance Gap (F1-Score): {abs(f1_train - f1_test):.2%}")