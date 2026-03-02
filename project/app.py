import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. تحميل البيانات
# تأكد أن ملف الاكسل بنفس الاسم وموجود في نفس مجلد المشروع
try:
    df = pd.read_excel('Real estate valuation data set.xlsx')
except FileNotFoundError:
    print("Error: الملف غير موجود")
    exit()


print("="*50)
print("PART 2: DATA PREPROCESSING & FEATURE SELECTION")
print("="*50)

# 2. نظرة أولية على البيانات (أول 5 صفوف)
print("\n[1] First 5 rows of the dataset:")
print(df.head())

# 3. التأكد من القيم المفقودة (Missing Values)
print("\n[2] Checking for Missing Values:")
null_counts = df.isnull().sum()
print(null_counts)

# 4. حذف الأعمدة غير الضرورية (Feature Selection)
if 'No' in df.columns:
    df.drop(columns=['No'], inplace=True)
    print("\n[3] Feature 'No' has been dropped.")


# حذف عمود تاريخ المعاملة (X1) إذا وُجد
if 'X1 transaction date' in df.columns:
    df.drop(columns=['X1 transaction date'], inplace=True)
    print("\n Feature 'X1 transaction date' has been dropped.")



print("\n[4] Correlation Matrix (Relationship with House Price):")

#5. بنركزوا على علاقة كل الميزات بالسعر (Y)
correlation = df.corr()['Y house price of unit area'].sort_values(ascending=False)
print(correlation)

#  تقسيم البيانات إلى مدخلات (X) وهدف (y)
X = df.drop(columns=['Y house price of unit area'])
y = df['Y house price of unit area']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[5] Dataset Splitting & Scaling:")
print(f"Total instances: {len(df)}")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print("-" * 50)
print("Preprocessing Completed Successfully!")
print("="*50)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

print("="*50)
print("PART 3: MODEL BUILDING & COMPARISON")
print("="*50)

# 1. تعريف النماذج
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)

# 2. تدريب النماذج (نستخدم البيانات المعالجة من Part 2)
lr_model.fit(X_train_scaled, y_train)
dt_model.fit(X_train_scaled, y_train)

# 3. التوقع على بيانات الاختبار
lr_pred = lr_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test_scaled)

# 4. حساب الدقة (R2 Score)
lr_r2 = r2_score(y_test, lr_pred) * 100
dt_r2 = r2_score(y_test, dt_pred) * 100

print(f"\n[1] Linear Regression Accuracy: {lr_r2:.2f}%")
print(f"[2] Decision Tree Accuracy:      {dt_r2:.2f}%")

print("\n" + "-"*30)
print("FINAL COMPARISON TABLE")
print("-"*30)
print(f"{'Model':<20} | {'R2 Score':<10}")
print(f"{'-'*20}-|-{'-'*10}")
print(f"{'Linear Regression':<20} | {lr_r2:.2f}%")
print(f"{'Decision Tree':<20} | {dt_r2:.2f}%")
print("-"*30)

# 5. استنتاج بسيط يطبع في الـ Terminal
if dt_r2 > lr_r2:
    best_model = "Decision Tree"
else:
    best_model = "Linear Regression"

print(f"\nConclusion: The {best_model} is the best model for this dataset.")
print("="*50)

from sklearn.metrics import mean_squared_error, r2_score

# 1. حساب المقاييس (الـ R2 والـ MSE) لكل مودل
lr_r2 = r2_score(y_test, lr_pred) * 100
dt_r2 = r2_score(y_test, dt_pred) * 100

lr_mse = mean_squared_error(y_test, lr_pred)
dt_mse = mean_squared_error(y_test, dt_pred)

print("="*60)
print("PART 4: MODEL EVALUATION & COMPARISON")
print("="*60)

# 2. طباعة المقاييس التفصيلية لكل مودل (Metrics for each model)
print(f"[1] Linear Regression Metrics:")
print(f"    - R-squared (R2): {lr_r2:.2f}%")
print(f"    - Mean Squared Error (MSE): {lr_mse:.2f}")

print(f"\n[2] Decision Tree Metrics:")
print(f"    - R-squared (R2): {dt_r2:.2f}%")
print(f"    - Mean Squared Error (MSE): {dt_mse:.2f}")




# 3. جدول المقارنة النهائي (Comparison Table)
# توا أضفنا عمود الـ MSE للجدول باش يكون التقرير كامل
print("\n" + "-"*55)
print(f"{'Model':<20} | {'R2 Score':<12} | {'MSE':<10}")
print("-" * 55)
print(f"{'Linear Regression':<20} | {lr_r2:.2f}%{'':<5} | {lr_mse:.2f}")
print(f"{'Decision Tree':<20} | {dt_r2:.2f}%{'':<5} | {dt_mse:.2f}")
print("-" * 55)





# 4. الخلاصة النهائية (Final Conclusion)
print("\nFinal Conclusion:")
print("The Linear Regression is the best model for this project.")
print("It achieved the highest accuracy (R2) and the lowest error (MSE),")
print("proving it is more reliable for predicting house prices.")
print("="*60)