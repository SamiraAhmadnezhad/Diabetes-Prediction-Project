# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.tree import DecisionTreeClassifier , plot_tree
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import seaborn as sns
# import matplotlib.pyplot as plt

 
# URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#                 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# data = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
 
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# data[columns_to_impute] = data[columns_to_impute].replace(0, np.nan)
# for col in columns_to_impute:
#     data[col].fillna(data.groupby('Outcome')[col].transform('mean'), inplace=True)

 
# X_full = data.drop('Outcome', axis=1)
# y_full = data['Outcome']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_full)
 

# k = 10 # تعداد همسایگان
# knn = NearestNeighbors(n_neighbors=k)
# knn.fit(X_scaled)
# distances, _ = knn.kneighbors(X_scaled)
 
# outlier_scores = distances.mean(axis=1)
 
# threshold = np.percentile(outlier_scores, 95)
# mask = outlier_scores < threshold

 
# X_clean = X_full[mask]
# y_clean = y_full[mask]
 
# X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.325, random_state=42, stratify=y_clean)
 
# model = DecisionTreeClassifier(random_state=42)
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [3, 4, 5, 6, 7, 8], # max depth is used for preventing the dicision tree from overfitting 
#     'min_samples_split': [9,10,11,12,13,14,15],
#     'min_samples_leaf': [6,7,8]
# }

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)

 
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# final_accuracy = accuracy_score(y_test, y_pred)

# print(f"best parameters: {grid_search.best_params_}")
# print(f"Accuracy: {final_accuracy * 100:.2f}%")

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Salamat (0)', 'Diabeti (1)'],
#             yticklabels=['Salamat (0)', 'Diabeti (1)'])
# plt.xlabel('Pishbini Shodeh')
# plt.ylabel('Vaghei')
# plt.title('Confusion Matrix (KNN-Out حذف Outlier)')
# plt.show() 

# plt.figure(figsize=(20, 10))
# plot_tree(best_model, 
#           feature_names=X_full.columns,
#           class_names=['Salamat (0)', 'Diabeti (1)'],
#           filled=True, 
#           rounded=True, 
#           fontsize=10)
# plt.title("Visualization of the Best Decision Tree")
# plt.show()

#####################################################################################################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import seaborn as sns
# import matplotlib.pyplot as plt

# # --- خواندن داده‌ها ---
# URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#                 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# data = pd.read_csv(URL, header=None, names=COLUMN_NAMES)

# # --- جدا کردن X و y ---
# X_full = data.drop('Outcome', axis=1)
# y_full = data['Outcome']

# # --- تقسیم داده‌ها به train/test ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X_full, y_full, test_size=0.325, random_state=42, stratify=y_full
# )

# # --- پیش‌پردازش: جایگزینی 0 با NaN و پر کردن با میانگین هر کلاس ---
# columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# # فقط روی train
# X_train[columns_to_impute] = X_train[columns_to_impute].replace(0, np.nan)
# for col in columns_to_impute:
#     X_train[col].fillna(X_train.groupby(y_train)[col].transform('mean'), inplace=True)

# # روی test هم همان مقادیر از train را استفاده می‌کنیم
# for col in columns_to_impute:
#     mean_values = X_train.groupby(y_train)[col].mean()
#     X_test[col] = X_test[col].replace(0, np.nan)
#     for cls in mean_values.index:
#         X_test.loc[(y_test == cls) & (X_test[col].isna()), col] = mean_values[cls]

# # --- حذف Outlier ها فقط با استفاده از train ---
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# k = 10
# knn = NearestNeighbors(n_neighbors=k)
# knn.fit(X_train_scaled)
# distances, _ = knn.kneighbors(X_train_scaled)

# outlier_scores = distances.mean(axis=1)
# threshold = np.percentile(outlier_scores, 95)
# mask_train = outlier_scores < threshold

# X_train_clean = X_train[mask_train]
# y_train_clean = y_train[mask_train]

# # --- آموزش مدل با GridSearchCV ---
# model = DecisionTreeClassifier(random_state=42)
# param_grid = {
#     'criterion': ['gini'],
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'min_samples_split': [9, 10, 11, 12, 13, 14, 15],
#     'min_samples_leaf': [6, 7, 8]
# }

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
#                            scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train_clean, y_train_clean)

# # --- ارزیابی ---
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# final_accuracy = accuracy_score(y_test, y_pred)
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Accuracy: {final_accuracy * 100:.2f}%")

# # --- ماتریس درهم‌ریختگی ---
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Salamat (0)', 'Diabeti (1)'],
#             yticklabels=['Salamat (0)', 'Diabeti (1)'])
# plt.xlabel('Pishbini Shodeh')
# plt.ylabel('Vaghei')
# plt.title('Confusion Matrix (حذف Outlier با Train)')
# plt.show()

# # --- رسم درخت تصمیم ---
# plt.figure(figsize=(20, 10))
# plot_tree(best_model,
#           feature_names=X_full.columns,
#           class_names=['Salamat (0)', 'Diabeti (1)'],
#           filled=True,
#           rounded=True,
#           fontsize=10)
# plt.title("Visualization of the Best Decision Tree")
# plt.show()
##################################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt

# --- خواندن داده‌ها ---
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(URL, header=None, names=COLUMN_NAMES)

# --- جدا کردن X و y ---
X_full = data.drop('Outcome', axis=1)
y_full = data['Outcome']

# --- تقسیم داده‌ها به train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.325, random_state=42, stratify=y_full
)

# --- پیش‌پردازش: جایگزینی 0 با NaN و پر کردن با میانگین هر کلاس ---
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

X_train = X_train.copy()
X_test = X_test.copy()

X_train[columns_to_impute] = X_train[columns_to_impute].replace(0, np.nan)
for col in columns_to_impute:
    means = X_train.groupby(y_train)[col].transform('mean')
    X_train[col] = X_train[col].fillna(means)

for col in columns_to_impute:
    mean_values = X_train.groupby(y_train)[col].mean()
    X_test[col] = X_test[col].replace(0, np.nan)
    for cls in mean_values.index:
        mask = (y_test == cls) & (X_test[col].isna())
        X_test.loc[mask, col] = mean_values[cls]

# --- حذف Outlier ها فقط با استفاده از train ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 10
knn = NearestNeighbors(n_neighbors=k)
knn.fit(X_train_scaled)
distances, _ = knn.kneighbors(X_train_scaled)

outlier_scores = distances.mean(axis=1)
threshold = np.percentile(outlier_scores, 95)
mask_train = outlier_scores < threshold

X_train_clean = X_train[mask_train]
y_train_clean = y_train[mask_train]

# --- آموزش مدل با GridSearchCV (class_weight='balanced') ---
model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [9, 10, 11, 12, 13, 14, 15],
    'min_samples_leaf': [6, 7, 8]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_clean, y_train_clean)

# --- ارزیابی ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

train_accuracy = best_model.score(X_train, y_train)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {final_accuracy * 100:.2f}%\n")

# --- گزارش کامل ---
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Salamat (0)', 'Diabeti (1)']))

# --- اهمیت ویژگی‌ها ---
importances = best_model.feature_importances_
features = X_full.columns

feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance in Decision Tree')
plt.show()

# --- ماتریس درهم‌ریختگی ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Salamat (0)', 'Diabeti (1)'],
            yticklabels=['Salamat (0)', 'Diabeti (1)'])
plt.xlabel('Pishbini Shodeh')
plt.ylabel('Vaghei')
plt.title('Confusion Matrix (حذف Outlier + class_weight balanced)')
plt.show()

# --- رسم درخت تصمیم ---
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=X_full.columns,
          class_names=['Salamat (0)', 'Diabeti (1)'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Visualization of the Best Decision Tree")
plt.show()

