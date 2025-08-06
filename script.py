import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt

 
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
 
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_impute] = data[columns_to_impute].replace(0, np.nan)
for col in columns_to_impute:
    data[col].fillna(data.groupby('Outcome')[col].transform('mean'), inplace=True)

 
X_full = data.drop('Outcome', axis=1)
y_full = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
 

k = 10 # تعداد همسایگان
knn = NearestNeighbors(n_neighbors=k)
knn.fit(X_scaled)
distances, _ = knn.kneighbors(X_scaled)
 
outlier_scores = distances.mean(axis=1)
 
threshold = np.percentile(outlier_scores, 95)
mask = outlier_scores < threshold

 
X_clean = X_full[mask]
y_clean = y_full[mask]
 
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.325, random_state=42, stratify=y_clean)
 
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8], # max depth is used for preventing the dicision tree from overfitting 
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

 
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"best parameters: {grid_search.best_params_}")
print(f"Accuracy: {final_accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Salamat (0)', 'Diabeti (1)'],
            yticklabels=['Salamat (0)', 'Diabeti (1)'])
plt.xlabel('Pishbini Shodeh')
plt.ylabel('Vaghei')
plt.title('Confusion Matrix (KNN-Out حذف Outlier)')
plt.show() 

plt.figure(figsize=(20, 10))
plot_tree(best_model, 
          feature_names=X_full.columns,
          class_names=['Salamat (0)', 'Diabeti (1)'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Visualization of the Best Decision Tree")
plt.show()
