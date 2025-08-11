# ایمپورت کردن کتابخانه مورد نیاز
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import GridSearchCV

# آدرس فایل دیتاست Pima در سیستم شما
# این آدرس را با آدرس فایل خودتان جایگزین کنید
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['pregnant', 'glucose', 'pressure', 'triceps', 'insulin', 'mass', 'pedigree', 'age', 'diabetes']
df = pd.read_csv(url, header=None, names=column_names)
# --- بررسی‌های اولیه ---

# ۱. نمایش ابعاد دیتاست (تعداد سطرها و ستون‌ها)
print("ابعاد دیتاست:")
print(df.shape)

# ۲. نمایش اطلاعات کلی دیتاست شامل نوع داده هر ستون و تعداد مقادیر غیرنال
print("\nاطلاعات کلی و نوع داده‌ها:")
print(df.info())

# ۳. نمایش ۵ ردیف اول دیتاست برای آشنایی با ساختار داده‌ها
print("\n۵ ردیف اول دیتاست:")
print(df.head())
#################################################################
#2
# --- شناسایی و جایگزینی داده‌های غیرممکن ---

# ستون‌هایی که مقدار صفر در آن‌ها بی‌معنی است (با نام‌های صحیح دیتافریم شما)
# توجه کنید که از 'triceps' به جای 'SkinThickness' و 'mass' به جای 'BMI' استفاده شده است
cols_to_replace_zero = ['glucose', 'pressure', 'triceps', 'insulin', 'mass']

# جایگزینی مقدار 0 با NaN (Not a Number) در ستون‌های مشخص شده
df.loc[:, cols_to_replace_zero] = df.loc[:, cols_to_replace_zero].replace(0, np.nan)


# --- شمارش تعداد مقادیر گمشده ---

# شمارش تعداد مقادیر NaN در هر ستون
missing_values_count = df.isnull().sum()

print("تعداد مقادیر گمشده (NaN) در هر ستون پس از جایگزینی صفرها:")
print(missing_values_count)
#########################################################
#3
import matplotlib.pyplot as plt
import seaborn as sns

# --- ۱. آمار توصیفی ---
# محاسبه آمار توصیفی برای ستون‌های عددی
# .T برای نمایش بهتر (ترانهاده کردن) خروجی است
print("--- آمار توصیفی داده‌ها ---")
print(df.describe().T)


# --- ۲. بررسی توزیع کلاس هدف (diabetes) ---
print("\n--- توزیع کلاس هدف (دیابت) ---")
print(df['diabetes'].value_counts())

# رسم نمودار برای توزیع کلاس هدف
plt.figure(figsize=(6, 4))
sns.countplot(x='diabetes', data=df)
plt.title('توزیع تعداد بیماران دیابتی و غیردیابتی')
plt.xlabel('وضعیت دیابت (0: سالم, 1: دیابتی)')
plt.ylabel('تعداد')
plt.show()


# --- ۳. توزیع هر یک از ویژگی‌ها ---
print("\n--- رسم نمودارهای توزیع ویژگی‌ها (Histogram) ---")
# رسم هیستوگرام برای هر ویژگی
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('هیستوگرام توزیع مقادیر برای هر ویژگی')
plt.tight_layout(rect=[0, 0, 1, 0.96]) # برای جلوگیری از همپوشانی عناوین
plt.show()
#################################################

# =================================================================
# گام ۴ (جدید): جدا کردن داده‌ها و سپس جایگزینی مقادیر گمشده
# =================================================================

# --- ۴.۱. جدا کردن ویژگی‌ها (X) از متغیر هدف (y) ---
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# --- ۴.۲. تقسیم داده‌ها به آموزش و آزمون (قبل از هر کار دیگری) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# نکته: stratify=y تضمین می‌کند که نسبت کلاس‌ها در مجموعه آموزش و آزمون یکسان باشد. این برای داده‌های نامتوازن بسیار مفید است.

# --- ۴.۳. جایگزینی مقادیر گمشده (Imputation) به روش صحیح ---
# محاسبه میانه فقط از داده‌های آموزشی
median_values = X_train.median()
print("--- میانه محاسبه شده از داده‌های آموزشی ---")
print(median_values)

# پر کردن مقادیر گمشده در هر دو مجموعه با استفاده از میانه آموزشی
X_train = X_train.fillna(median_values)
X_test = X_test.fillna(median_values)

print("\n--- بررسی مقادیر گمشده پس از جایگزینی ---")
print("مقادیر گمشده در X_train:", X_train.isnull().sum().sum())
print("مقادیر گمشده در X_test:", X_test.isnull().sum().sum())


# =================================================================
# گام ۵ (اختیاری اما بسیار پیشنهادی): استانداردسازی داده‌ها
# =================================================================
# الگوریتم‌های مبتنی بر درخت به استانداردسازی نیاز ندارند، اما این یک تمرین خوب است
# و برای بسیاری از الگوریتم‌های دیگر ضروری است.
scaler = StandardScaler()

# scaler را فقط روی داده‌های آموزشی fit می‌کنیم
X_train_scaled = scaler.fit_transform(X_train)
# داده‌های آزمون را فقط transform می‌کنیم (برای جلوگیری از نشت داده)
X_test_scaled = scaler.transform(X_test)


# =================================================================
# گام ۶ (پیشرفته): متوازن کردن داده‌ها با SMOTE و ساخت مدل نهایی
# =================================================================
print("\n--- وضعیت کلاس‌ها در y_train قبل از SMOTE ---")
print(sorted(Counter(y_train).items()))

smote = SMOTE(random_state=42)
# ما از داده‌های مقیاس‌شده (scaled) برای SMOTE استفاده می‌کنیم
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\n--- وضعیت کلاس‌ها بعد از SMOTE ---")
print(sorted(Counter(y_train_resampled).items()))

# جستجوی شبکه‌ای برای Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'min_samples_leaf': [3, 5],
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced') # class_weight هم به تعادل کمک می‌کند
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='f1', n_jobs=-1, verbose=2)

print("\n--- در حال اجرای GridSearchCV برای Random Forest... ---")
grid_search_rf.fit(X_train_resampled, y_train_resampled)

print("\n--- بهترین پارامترهای پیدا شده برای Random Forest ---")
print(grid_search_rf.best_params_)

best_rf_model = grid_search_rf.best_estimator_


# =================================================================
# گام ۷ (نهایی و گسترش‌یافته): ارزیابی دوگانه مدل Random Forest
# =================================================================
print("\n\n--- بخش ۷: ارزیابی دوگانه مدل Random Forest ---")

# ۷.۱. ارزیابی روی داده‌های آزمون (Test Data)
print("\n================ ارزیابی عملکرد روی داده‌های آزمون (Test) ================")
# پیش‌بینی روی داده‌های آزمون مقیاس‌شده
y_pred_test_rf = best_rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_test_rf, target_names=['Salamat (0)', 'Diabeti (1)']))
accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
print(f"دقت کلی (Accuracy) روی داده‌های آزمون: {accuracy_test_rf:.2%}")

# ۷.۲. ارزیابی روی داده‌های آموزشی (Train Data) برای بررسی بیش‌برازش
# نکته مهم: ما باید روی داده‌های آموزشی *قبل* از SMOTE ارزیابی کنیم تا یک مقایسه منصفانه داشته باشیم،
# اما چون مدل روی داده‌های resampled آموزش دیده، ارزیابی روی داده‌های اصلی آموزشی مقیاس‌شده (X_train_scaled) انجام می‌شود.
print("\n================ ارزیابی عملکرد روی داده‌های آموزشی (Train) ================")
y_pred_train_rf = best_rf_model.predict(X_train_scaled)
print(classification_report(y_train, y_pred_train_rf, target_names=['Salamat (0)', 'Diabeti (1)']))
accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
print(f"دقت کلی (Accuracy) روی داده‌های آموزشی: {accuracy_train_rf:.2%}")

# ۷.۳. مقایسه و نتیجه‌گیری نهایی برای مدل Random Forest
print("\n================ مقایسه و نتیجه‌گیری برای مدل Random Forest ================")
f1_test_rf = classification_report(y_test, y_pred_test_rf, output_dict=True)['weighted avg']['f1-score']
f1_train_rf = classification_report(y_train, y_pred_train_rf, output_dict=True)['weighted avg']['f1-score']
print(f"امتیاز F1 نهایی مدل روی داده‌های دیده‌نشده (Test F1-Score): {f1_test_rf:.2%}")
print(f"امتیاز F1 مدل روی داده‌هایی که با آن آموزش دیده (Train F1-Score): {f1_train_rf:.2%}")

# محاسبه اختلاف
accuracy_diff = abs(accuracy_train_rf - accuracy_test_rf)
f1_diff = abs(f1_train_rf - f1_test_rf)
print(f"\nاختلاف دقت (Accuracy) بین آموزش و آزمون: {accuracy_diff:.2%}")
print(f"اختلاف امتیاز F1 بین آموزش و آزمون: {f1_diff:.2%}")

if accuracy_diff < 0.07 and f1_diff < 0.07:
    print("\nنتیجه: عملکرد مدل روی داده‌های آموزش و آزمون بسیار نزدیک است.")
    print("این یک نشانه قوی است که مدل دچار بیش‌برازش (Overfitting) نشده و نتایج آن قابل اعتماد است.")
else:
    print("\nنتیجه: اختلاف عملکرد مدل روی داده‌های آموزش و آزمون هنوز قابل توجه است.")
    print("این ممکن است نشانه میزانی از بیش‌برازش باشد.")