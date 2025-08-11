# ایمپورت کردن کتابخانه مورد نیاز
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
#4
# --- ۱. ایمپوت کردن (جایگزینی) مقادیر گمشده ---

# ما از میانه (median) برای پر کردن مقادیر NaN استفاده می‌کنیم
# چون میانه نسبت به داده‌های پرت مقاوم‌تر است و برخی ستون‌ها چولگی داشتند.

# .fillna() مقادیر NaN را با مقدار داده شده پر می‌کند.
# inplace=True تغییرات را مستقیماً روی دیتافریم اصلی اعمال می‌کند.
df['glucose'].fillna(df['glucose'].median(), inplace=True)
df['pressure'].fillna(df['pressure'].median(), inplace=True)
df['triceps'].fillna(df['triceps'].median(), inplace=True)
df['insulin'].fillna(df['insulin'].median(), inplace=True)
df['mass'].fillna(df['mass'].median(), inplace=True)


# --- بررسی مجدد برای اطمینان از حذف همه مقادیر گمشده ---
print("--- تعداد مقادیر گمشده پس از جایگزینی با میانه ---")
print(df.isnull().sum())
#####################################################
#5
from sklearn.model_selection import train_test_split

# --- جدا کردن ویژگی‌ها (X) از متغیر هدف (y) ---
# X شامل تمام ستون‌ها به جز 'diabetes' است
X = df.drop('diabetes', axis=1)
# y فقط شامل ستون 'diabetes' است
y = df['diabetes']


# --- تقسیم داده‌ها به آموزش و آزمون ---
# test_size=0.2 یعنی 20% داده‌ها برای آزمون و 80% برای آموزش
# random_state=42 یک عدد دلخواه برای اطمینان از این است که هر بار کد را اجرا می‌کنیم،
# تقسیم‌بندی به همین شکل انجام شود تا نتایج قابل تکرار باشند.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- نمایش ابعاد داده‌های تقسیم شده برای اطمینان ---
print("--- ابعاد داده‌های اصلی ---")
print("شکل X (ویژگی‌ها):", X.shape)
print("شکل y (هدف):", y.shape)

print("\n--- ابعاد داده‌های آموزش ---")
print("شکل X_train:", X_train.shape)
print("شکل y_train:", y_train.shape)

print("\n--- ابعاد داده‌های آزمون ---")
print("شکل X_test:", X_test.shape)
print("شکل y_test:", y_test.shape)
##################################################
#6

# --- ۱. تعریف فضای جستجو برای پارامترها ---
# ما به GridSearchCV می‌گوییم چه مقادیری را برای هر پارامتر امتحان کند.
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10],  # امتحان کردن عمق‌های مختلف
    'min_samples_leaf': [5, 10, 15, 20], # امتحان کردن حداقل نمونه‌های مختلف در هر برگ
    'criterion': ['gini', 'entropy']     # امتحان کردن هر دو معیار تقسیم
}

# --- ۲. ساخت مدل پایه و جستجوی شبکه‌ای ---
# یک درخت تصمیم خالی می‌سازیم
dt_model = DecisionTreeClassifier(random_state=42)

# GridSearchCV را تنظیم می‌کنیم
# cv=5 یعنی داده‌های آموزشی را به ۵ بخش تقسیم کرده و فرآیند را ۵ بار تکرار می‌کند تا نتیجه پایدار باشد.
# scoring='f1' یعنی بهترین مدل را بر اساس امتیاز F1 انتخاب کن (که برای داده‌های نامتوازن بهتر از accuracy است).
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)

# --- ۳. اجرای جستجو برای یافتن بهترین پارامترها ---
# این فرآیند ممکن است کمی زمان ببرد
grid_search.fit(X_train, y_train)

# --- ۴. نمایش بهترین پارامترهای پیدا شده ---
print("\n--- بهترین پارامترهای پیدا شده توسط GridSearchCV ---")
print(grid_search.best_params_)

# --- ۵. استخراج بهترین مدل ---
# GridSearchCV به طور خودکار بهترین مدل را با بهترین پارامترها دوباره روی کل داده‌های آموزشی، آموزش می‌دهد.
best_dt_model = grid_search.best_estimator_
print("\n--- مدل بهینه‌سازی شده با موفقیت ساخته شد ---")
######################################################
#7

# --- ۱. ارزیابی روی داده‌های آزمون (Test Data) ---
print("==========================================================")
print("              ارزیابی عملکرد روی داده‌های آزمون (Test)     ")
print("==========================================================")
# پیش‌بینی روی داده‌های آزمون با استفاده از مدل بهینه
y_pred_test = best_dt_model.predict(X_test)

# نمایش گزارش ارزیابی برای داده‌های آزمون
print("\n--- گزارش ارزیابی (Test) ---")
print(classification_report(y_test, y_pred_test, target_names=['Salamat (0)', 'Diabeti (1)']))

accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"\nدقت کلی (Accuracy) روی داده‌های آزمون: {accuracy_test:.2f}")


# --- ۲. ارزیابی روی داده‌های آموزشی (Train Data) ---
print("\n\n==========================================================")
print("              ارزیابی عملکرد روی داده‌های آموزشی (Train)    ")
print("==========================================================")
# پیش‌بینی روی داده‌های آموزشی برای بررسی بیش‌برازش
y_pred_train = best_dt_model.predict(X_train)

# نمایش گزارش ارزیابی برای داده‌های آموزشی
print("\n--- گزارش ارزیابی (Train) ---")
print(classification_report(y_train, y_pred_train, target_names=['Salamat (0)', 'Diabeti (1)']))

accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"\nدقت کلی (Accuracy) روی داده‌های آموزشی: {accuracy_train:.2f}")


# --- ۳. مقایسه و نتیجه‌گیری نهایی ---
print("\n\n==========================================================")
print("                    مقایسه و نتیجه‌گیری نهایی                 ")
print("==========================================================")
print(f"دقت نهایی مدل روی داده‌های دیده‌نشده (Test Accuracy): {accuracy_test:.2%}")
print(f"دقت مدل روی داده‌هایی که با آن آموزش دیده (Train Accuracy): {accuracy_train:.2%}")

f1_test = classification_report(y_test, y_pred_test, output_dict=True)['weighted avg']['f1-score']
f1_train = classification_report(y_train, y_pred_train, output_dict=True)['weighted avg']['f1-score']

print(f"امتیاز F1 نهایی مدل روی داده‌های دیده‌نشده (Test F1-Score): {f1_test:.2%}")
print(f"امتیاز F1 مدل روی داده‌هایی که با آن آموزش دیده (Train F1-Score): {f1_train:.2%}")

if abs(accuracy_train - accuracy_test) < 0.05 and abs(f1_train - f1_test) < 0.05:
    print("\nنتیجه: عملکرد مدل روی داده‌های آموزش و آزمون بسیار نزدیک است.")
    print("این یک نشانه قوی است که مدل دچار بیش‌برازش (Overfitting) نشده و نتایج آن قابل اعتماد است.")
else:
    print("\nنتیجه: اختلاف عملکرد مدل روی داده‌های آموزش و آزمون قابل توجه است.")
    print("این ممکن است نشانه میزانی از بیش‌برازش باشد. مدل هنوز جای بهبود دارد.")