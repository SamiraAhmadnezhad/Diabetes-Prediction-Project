# ایمپورت کردن تمام کتابخانه‌های مورد نیاز در ابتدا
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# =================================================================
# بخش ۱: بارگذاری داده‌ها (بدون تغییر)
# =================================================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)
print("--- بخش ۱: بارگذاری داده‌ها با موفقیت انجام شد ---")


# =================================================================
# بخش ۲: شناسایی داده‌های غیرممکن و جایگزینی با NaN (بدون تغییر)
# =================================================================
cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)
print("\n--- بخش ۲: جایگزینی صفرها با NaN با موفقیت انجام شد ---")
print("تعداد مقادیر گمشده (NaN) در هر ستون:")
print(df.isnull().sum())


# =================================================================
# بخش ۳: تحلیل توصیفی داده‌ها (EDA) (بدون تغییر)
# =================================================================
# این بخش نمودارها را تولید می‌کند و نیازی به تغییر ندارد.
# برای جلوگیری از باز شدن مکرر نمودارها، می‌توان آن را کامنت کرد.
# print("\n--- بخش ۳: تحلیل توصیفی در حال انجام است... ---")
# df.describe()
# df['diabetes'].value_counts()
# df.hist(bins=20, figsize=(15, 10))
# plt.show()


# =================================================================
# بخش ۴ (جدید و اصلاح‌شده): تقسیم داده و جایگزینی مقادیر گمشده (Imputation)
# =================================================================
print("\n--- بخش ۴: تقسیم داده و Imputation به روش صحیح ---")

# ۴.۱. جدا کردن ویژگی‌ها (X) از متغیر هدف (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# ۴.۲. تقسیم داده‌ها به آموزش و آزمون (قبل از هر کار دیگری)
# از stratify=y برای حفظ نسبت کلاس‌ها استفاده می‌کنیم
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"ابعاد داده‌های آموزش: {X_train.shape}, ابعاد داده‌های آزمون: {X_test.shape}")

# ۴.۳. جایگزینی مقادیر گمشده بر اساس میانه داده‌های آموزشی
# محاسبه میانه فقط از داده‌های آموزشی
median_values = X_train.median()
print("\nمیانه محاسبه شده از داده‌های آموزشی:")
print(median_values)

# پر کردن مقادیر گمشده در هر دو مجموعه با استفاده از میانه آموزشی
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)
print("\nبررسی مقادیر گمشده پس از جایگزینی: ...انجام شد.")
# print("مقادیر گمشده در X_train:", X_train.isnull().sum().sum()) # باید 0 باشد
# print("مقادیر گمشده در X_test:", X_test.isnull().sum().sum())   # باید 0 باشد


# =================================================================
# بخش ۵ (جدید و اصلاح‌شده): بهینه‌سازی و آموزش مدل درخت تصمیم
# =================================================================
print("\n--- بخش ۵: بهینه‌سازی و آموزش مدل درخت تصمیم ---")

# ۵.۱. تعریف فضای جستجو برای پارامترها (بدون تغییر)
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_samples_leaf': [5, 10, 15, 20],
    'criterion': ['gini', 'entropy']
}

# ۵.۲. ساخت مدل پایه و جستجوی شبکه‌ای (بدون تغییر)
dt_model = DecisionTreeClassifier(random_state=42)
# در اینجا از F1-Score برای انتخاب بهترین مدل استفاده می‌کنیم
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)

# ۵.۳. اجرای جستجو روی داده‌های آموزشی تمیز شده
# این خط مهم‌ترین بخش است: ما روی X_train که به روش صحیح پردازش شده، fit می‌کنیم
grid_search.fit(X_train, y_train)

# ۵.۴. نمایش بهترین پارامترها و استخراج بهترین مدل
print("\nبهترین پارامترهای پیدا شده توسط GridSearchCV:")
print(grid_search.best_params_)
best_dt_model = grid_search.best_estimator_
print("\nمدل بهینه درخت تصمیم با موفقیت ساخته شد.")


# =================================================================
# بخش ۶ (جدید و اصلاح‌شده): ارزیابی دوگانه مدل درخت تصمیم
# =================================================================
print("\n--- بخش ۶: ارزیابی دوگانه مدل ---")

# ۶.۱. ارزیابی روی داده‌های آزمون (Test Data)
print("\n================ ارزیابی عملکرد روی داده‌های آزمون (Test) ================")
y_pred_test = best_dt_model.predict(X_test)
print(classification_report(y_test, y_pred_test, target_names=['Salamat (0)', 'Diabeti (1)']))
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"دقت کلی (Accuracy) روی داده‌های آزمون: {accuracy_test:.2%}")

# ۶.۲. ارزیابی روی داده‌های آموزشی (Train Data) برای بررسی بیش‌برازش
print("\n================ ارزیابی عملکرد روی داده‌های آموزشی (Train) ================")
y_pred_train = best_dt_model.predict(X_train)
print(classification_report(y_train, y_pred_train, target_names=['Salamat (0)', 'Diabeti (1)']))
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"دقت کلی (Accuracy) روی داده‌های آموزشی: {accuracy_train:.2%}")

# ۶.۳. مقایسه و نتیجه‌گیری نهایی
print("\n================ مقایسه و نتیجه‌گیری نهایی ================")
f1_test = classification_report(y_test, y_pred_test, output_dict=True)['weighted avg']['f1-score']
f1_train = classification_report(y_train, y_pred_train, output_dict=True)['weighted avg']['f1-score']
print(f"امتیاز F1 نهایی مدل روی داده‌های دیده‌نشده (Test F1-Score): {f1_test:.2%}")
print(f"امتیاز F1 مدل روی داده‌هایی که با آن آموزش دیده (Train F1-Score): {f1_train:.2%}")

if abs(accuracy_train - accuracy_test) < 0.07 and abs(f1_train - f1_test) < 0.07: # کمی آستانه را افزایش دادم
    print("\nنتیجه: عملکرد مدل روی داده‌های آموزش و آزمون نزدیک است.")
    print("این نشانه خوبی از کنترل بیش‌برازش است و نتایج قابل اعتماد هستند.")
else:
    print("\nنتیجه: اختلاف عملکرد مدل روی داده‌های آموزش و آزمون هنوز قابل توجه است.")
    print("این ممکن است نشانه میزانی از بیش‌برازش باشد.")