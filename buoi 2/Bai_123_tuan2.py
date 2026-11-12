# %% [markdown]
# # **BÀI 1**

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tạo tất cả biểu đồ trước, chỉ hiển thị một lần ở cuối file.
# Không gọi plt.show() trong từng phần vì nó sẽ chặn chương trình.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)

# %%
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)

X = diabetes.data.copy()
y = diabetes.target.copy()

df = X.copy()
df["target"] = y

# %%
print(X.head())

# %% [markdown]
# # **Tạo 1 bảng mới với 3 Features trong bộ dữ liệu Diabetes**

# %%
X_selected = X[['age', 'sex', 'bmi']]

print(X_selected.head())

# %% [markdown]
# # **dùng histogram để quan sát phân phối của cột AGE**

# %%
plt.figure(figsize=(7, 5))
plt.hist(X_selected['age'], bins=20)
plt.xlabel('age')
plt.ylabel('Frequency')   #số lượng mẫu xuất hiện trong một khoảng giá trị
plt.title('Distribution of age')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét age**: Dữ liệu tập trung nhiều ở vùng khoảng **0 đến 0.05**. Phân phối nhìn chung **tương đối cân bằng**, không có dấu hiệu lệch mạnh về một phía.

# %% [markdown]
# # **NHẬN XÉT PHÂN PHỐI VỀ SEX**

# %%
X_selected['sex'].unique()

# %%
X_selected['sex'].value_counts()

# %%
plt.figure(figsize=(7, 5))
plt.hist(X_selected['sex'], bins=2)
plt.xlabel('sex')
plt.ylabel('Frequency')
plt.title('Distribution of sex')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét phân phối sex:** Feature sex chỉ có **2 giá trị**. Hai nhóm có số lượng lần lượt là** 235 và 207 mẫu**, vì vậy phân phối giữa hai nhóm **tương đối cân bằng.**

# %%
plt.figure(figsize=(7, 5))
plt.hist(X_selected['bmi'], bins=20)
plt.xlabel('bmi')
plt.ylabel('Frequency')
plt.title('Distribution of bmi')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét bmi**: Dữ liệu **tập trung nhiều ở khoảng -0.05 đến 0.00**. Phân phối có **xu hướng lệch phải**, **do phía giá trị dương có phần đuôi kéo dài**.
#
# bmi được gọi là lệch phải — vì đuôi của phân phối kéo dài về bên phải

# %% [markdown]
# # **OUTLIER**

# %%
plt.figure(figsize=(7, 5))
plt.boxplot(X_selected['age'])
plt.ylabel('age')
plt.title('Boxplot of age')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# Nhận xét: age: Không phát hiện outlier trên Boxplot.

# %%
plt.figure(figsize=(7, 5))
plt.boxplot(X_selected['sex'])
plt.ylabel('sex')
plt.title('Boxplot of sex')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét: sex: Không phát hiện outlier trên Boxplot. Tuy nhiên, sex là biến chỉ có 2 mức giá trị nên việc đánh giá outlier bằng Boxplot không có nhiều ý nghĩa như đối với các feature liên tục như age và bmi.**

# %%
plt.figure(figsize=(7, 5))
plt.boxplot(X_selected['bmi'])
plt.ylabel('bmi')
plt.title('Boxplot of bmi')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét:
# bmi: Có 3 giá trị outlier ở phía trên. Các giá trị này lớn hơn đáng kể so với phần lớn dữ liệu.**

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X_selected['age'], y)
plt.xlabel('age')
plt.ylabel('target')
plt.title('Scatter Plot of age and target')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét: age và target có tương quan dương, nhưng tương quan không mạnh vì các điểm dữ liệu phân tán khá nhiều.**

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X_selected['sex'], y)
plt.xlabel('sex')
plt.ylabel('target')
plt.title('Scatter Plot of sex and target')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét: sex được chia thành 2 nhóm giá trị. Target của hai nhóm phân tán và chồng lấn khá nhiều, nên chưa thấy mối quan hệ rõ ràng giữa sex và target.**

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X_selected['bmi'], y)
plt.xlabel('bmi')
plt.ylabel('target')
plt.title('Scatter Plot of bmi and target')
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# **Nhận xét: bmi và target có tương quan dương và mối quan hệ này rõ hơn so với age và target.**

# %%
df_corr = X_selected.copy()
df_corr['target'] = y
print(df_corr.head())

# %%
corr_matrix = df_corr.corr()
print(corr_matrix)

# %% [markdown]
# **Nhận xét: Trong 3 đặc trưng age, sex, bmi, bmi có hệ số tương quan với target lớn nhất, khoảng 0.59. Điều này cho thấy bmi có mối tương quan dương rõ hơn với target, nên em ưu tiên bmi hơn nếu dựa trên phân tích tương quan.**

# %%
# Vẽ ma trận tương quan
plt.figure(figsize=(9, 7))

plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(
            j, i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Ma trận tương quan")
plt.tight_layout()
# Biểu đồ được giữ lại để hiển thị đồng thời ở cuối file.

# %% [markdown]
# ## Bài 2: Huấn luyện mô hình với các bộ feature khác nhau

# %%
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
# Model 1: Sử dụng feature bmi

# Tách bộ dữ liệu
X_bmi = df[["bmi"]]
y = df["target"]

X_train_bmi, X_test_bmi, y_train_bmi, y_test_bmi = train_test_split(
    X_bmi,
    y,
    test_size=0.2,
    random_state=42
)

# Khai báo mô hình
bmi_model = LinearRegression()

# Huấn luyện mô hình
bmi_model.fit(X_train_bmi, y_train_bmi)

# Dự đoán mô hình
bmi_pred = bmi_model.predict(X_test_bmi)

# Đánh giá mô hình
bmi_mae = mean_absolute_error(y_test_bmi, bmi_pred)
bmi_mse = mean_squared_error(y_test_bmi, bmi_pred)
bmi_rmse = np.sqrt(bmi_mse)
bmi_r2 = r2_score(y_test_bmi, bmi_pred)

print(f"BMI MAE  = {bmi_mae:.4f}")
print(f"BMI RMSE = {bmi_rmse:.4f}")
print(f"BMI R²   = {bmi_r2:.4f}")

# %%
# Model 2: Sử dụng feature bmi + bp

# Tách bộ dữ liệu
X_bmi_bp = df[["bmi", "bp"]]
y = df["target"]

X_train_bmi_bp, X_test_bmi_bp, y_train_bmi_bp, y_test_bmi_bp = train_test_split(
    X_bmi_bp,
    y,
    test_size=0.2,
    random_state=42
)
# Khai báo mô hình
bmi_bp_model = LinearRegression()

# Huấn luyện mô hình
bmi_bp_model.fit(X_train_bmi_bp, y_train_bmi_bp)

# Dự đoán mô hình
bmi_bp_pred = bmi_bp_model.predict(X_test_bmi_bp)
# Đánh giá mô hình bmi + bp
bmi_bp_mae = mean_absolute_error(y_test_bmi_bp, bmi_bp_pred)
bmi_bp_mse = mean_squared_error(y_test_bmi_bp, bmi_bp_pred)
bmi_bp_rmse = np.sqrt(bmi_bp_mse)
bmi_bp_r2 = r2_score(y_test_bmi_bp, bmi_bp_pred)

print(f"BMI + BP MAE = {bmi_bp_mae:.4f}")
print(f"BMI + BP RMSE = {bmi_bp_rmse:.4f}")
print(f"BMI + BP R² = {bmi_bp_r2:.4f}")

# %% [markdown]
# **Nhận xét: Mô hình sử dụng *bmi + bp* tốt hơn mô hình chỉ sử dụng *bmi*, vì MAE và RMSE đều giảm, đồng thời R² *tăng từ 0.2334 lên 0.2940*. Điều này cho thấy việc bổ sung feature bp đã giúp mô hình dự đoán target *tốt hơn*.**

# %%
# Model 3: Sử dụng Top 5 features theo tương quan

# Tìm Top 5 features có tương quan mạnh nhất với target
abs_corr = df.corr(numeric_only=True)["target"].drop("target").abs()

top_features = abs_corr.sort_values(ascending=False).head(5).index.tolist()

print("5 feature có |correlation| cao nhất với target:")
print(top_features)

# %%
# Tách bộ dữ liệu
X_top5 = df[top_features]
y = df["target"]

X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(
    X_top5,
    y,
    test_size=0.2,
    random_state=42
)


# Khai báo mô hình
top5_model = LinearRegression()

# Huấn luyện mô hình
top5_model.fit(X_train_top5, y_train_top5)

# Dự đoán mô hình
top5_pred = top5_model.predict(X_test_top5)


# Đánh giá mô hình
top5_mae = mean_absolute_error(y_test_top5, top5_pred)
top5_mse = mean_squared_error(y_test_top5, top5_pred)
top5_rmse = np.sqrt(top5_mse)
top5_r2 = r2_score(y_test_top5, top5_pred)

print(f"TOP 5 MAE = {top5_mae:.4f}")
print(f"TOP 5 RMSE = {top5_rmse:.4f}")
print(f"TOP 5 R² = {top5_r2:.4f}")

# %% [markdown]
# Nhận Xét: **Mô hình sử dụng *Top 5 features* *tốt hơn* hai mô hình trước *vì MAE và RMSE đều giảm*, trong khi* R² tăng*. Điều này cho thấy khi sử dụng các feature có *tương quan mạnh với target*, khả năng dự đoán của mô hình được cải thiện.**

# %%
# Model 4: Sử dụng toàn bộ 10 features

# Tách bộ dữ liệu
X_all = df.drop(columns=["target"])               #bỏ cột target
y = df["target"]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all,
    y,
    test_size=0.2,
    random_state=42
)

# Khai báo mô hình
all_model = LinearRegression()

# Huấn luyện mô hình
all_model.fit(X_train_all, y_train_all)

# Dự đoán mô hình
all_pred = all_model.predict(X_test_all)

# Đánh giá mô hình
all_mae = mean_absolute_error(y_test_all, all_pred)
all_mse = mean_squared_error(y_test_all, all_pred)
all_rmse = np.sqrt(all_mse)
all_r2 = r2_score(y_test_all, all_pred)

print(f"ALL FEATURES MAE = {all_mae:.4f}")
print(f"ALL FEATURES RMSE = {all_rmse:.4f}")
print(f"ALL FEATURES R² = {all_r2:.4f}")

# %% [markdown]
# **Nhận xét: Mô hình sử dụng toàn bộ 10 features cho kết quả tốt nhất trong 4 mô hình vì MAE và RMSE thấp nhất, đồng thời R² cao nhất. Điều này cho thấy khi sử dụng toàn bộ 10 features, mô hình dự đoán target tốt hơn so với các bộ feature còn lại.**

# %%
# So sánh 4 mô hình

comparison = pd.DataFrame({
    "Model": ["BMI", "BMI + BP", "Top 5 Features", "All 10 Features"],
    "MAE": [bmi_mae, bmi_bp_mae, top5_mae, all_mae],
    "RMSE": [bmi_rmse, bmi_bp_rmse, top5_rmse, all_rmse],
    "R²": [bmi_r2, bmi_bp_r2, top5_r2, all_r2]
})

print("BẢNG SO SÁNH MÔ HÌNH:")
print(comparison.to_string(index=False))

# %% [markdown]
# **Nhận xét: Khi tăng số lượng feature, kết quả của mô hình nhìn chung được cải thiện. Mô hình chỉ sử dụng bmi có kết quả thấp nhất. Khi bổ sung bp, MAE và RMSE giảm, R² tăng. Mô hình sử dụng Top 5 features tiếp tục cải thiện rõ rệt. Trong 4 mô hình, mô hình sử dụng toàn bộ 10 features cho kết quả tốt nhất với MAE = 42.7941, RMSE = 53.8534 và R² = 0.4526. Tuy nhiên, kết quả của mô hình 10 features chỉ tốt hơn Top 5 một lượng nhỏ.**

# %% [markdown]
# # Bài 3: Huấn luyện mô hình sử dụng bộ dữ liệu Diabetes bằng các thuật toán
# Ridge Regression = Hồi quy tuyến tính = dự đoán một giá trị số
#
# Decision Tree Regression, Support Vector
# Regression, Gradient Boosting Regression và so sánh với Linear
# Regression

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# %%
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# %%
# Linear Regression
linear_model = LinearRegression()

# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán trên tập test
linear_pred = linear_model.predict(X_test)

# %%
# Ridge Regression
ridge_model = Ridge()

# Huấn luyện mô hình
ridge_model.fit(X_train, y_train)

# Dự đoán trên tập test
ridge_pred = ridge_model.predict(X_test)

# %%
# Decision Tree Regression
tree_model = DecisionTreeRegressor(random_state=42)

# Huấn luyện mô hình
tree_model.fit(X_train, y_train)

# Dự đoán trên tập test
tree_pred = tree_model.predict(X_test)

# %%
# Support Vector Regression
svr_model = SVR()

# Huấn luyện mô hình
svr_model.fit(X_train, y_train)

# Dự đoán trên tập test
svr_pred = svr_model.predict(X_test)

# %%
# Gradient Boosting Regression
gb_model = GradientBoostingRegressor(random_state=42)

# Huấn luyện mô hình
gb_model.fit(X_train, y_train)

# Dự đoán trên tập test
gb_pred = gb_model.predict(X_test)

# %%
# Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

# %%
linear_mae, linear_rmse, linear_r2 = evaluate_model(y_test, linear_pred)
ridge_mae, ridge_rmse, ridge_r2 = evaluate_model(y_test, ridge_pred)
tree_mae, tree_rmse, tree_r2 = evaluate_model(y_test, tree_pred)
svr_mae, svr_rmse, svr_r2 = evaluate_model(y_test, svr_pred)
gb_mae, gb_rmse, gb_r2 = evaluate_model(y_test, gb_pred)

# %%
comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Decision Tree Regression",
        "Support Vector Regression",
        "Gradient Boosting Regression"
    ],
    "MAE": [
        linear_mae,
        ridge_mae,
        tree_mae,
        svr_mae,
        gb_mae
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        tree_rmse,
        svr_rmse,
        gb_rmse
    ],
    "R2": [
        linear_r2,
        ridge_r2,
        tree_r2,
        svr_r2,
        gb_r2
    ]
})

print("BẢNG SO SÁNH MÔ HÌNH:")
print(comparison.to_string(index=False))

# %% [markdown]
# Nhận xét:
# - Linear Regression có MAE thấp nhất.
# - Gradient Boosting Regression có RMSE thấp nhất và R² cao nhất.
# - Decision Tree Regression và Support Vector Regression cho kết quả kém hơn trong lần thử nghiệm này.
# - Xét tổng thể các chỉ số, Gradient Boosting Regression cho kết quả tốt nhất, tuy nhiên hiệu năng khá gần với Linear Regression.

# %%
models = {
    "Linear Regression": linear_model,
    "Ridge Regression": ridge_model,
    "Decision Tree Regression": tree_model,
    "Support Vector Regression": svr_model,
    "Gradient Boosting Regression": gb_model
}

for name, model in models.items():
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    print(name)
    print("R² Train:", train_r2)
    print("R² Test :", test_r2)
    print()

# %% [markdown]
# Nhận xét:
# - Linear Regression có MAE thấp nhất và kết quả giữa tập Train và Test khá ổn định.
# - Ridge Regression có kết quả Train và Test khá gần nhau nhưng hiệu năng thấp hơn Linear Regression.
# - Decision Tree Regression bị overfitting rõ rệt vì R² Train = 1 nhưng R² Test chỉ khoảng 0.061.
# - Support Vector Regression có R² thấp trên cả Train và Test nên hiệu năng chưa tốt.
# - Gradient Boosting Regression có RMSE thấp nhất và R² Test cao nhất, nhưng R² Train cao hơn Test khá nhiều nên có dấu hiệu overfitting.
# - Vì vậy, nếu xét cả hiệu năng trên tập Test và khả năng tổng quát hóa, Linear Regression là mô hình ổn định hơn trong thử nghiệm này.



# HIỂN THỊ TẤT CẢ CÁC BIỂU ĐỒ CÙNG LÚC
# Matplotlib mở đồng thời tất cả các Figure đã tạo ở trên.
plt.show()
