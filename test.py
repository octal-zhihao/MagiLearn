# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 加载训练数据集
train_df = pd.read_csv('data/train.csv')

# 获取数值特征列
num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns  # 数值特征
cat_cols = train_df.select_dtypes(include=['object']).columns  # 类别特征

# 加载测试数据集
test_df = pd.read_csv('data/test.csv')

# 进行缺失值处理
# 填充训练集和测试集的数值列
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
num_cols = num_cols[num_cols != 'SalePrice']
test_df[num_cols] = test_df[num_cols].fillna(train_df[num_cols].median())  # 使用训练集的中位数填充测试集

# 填充类别特征列
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])  # 使用训练集的众数填充测试集

# 确认缺失值已处理
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# 将类别特征进行 One-Hot 编码
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# 确保训练集和测试集具有相同的列
# 找出训练集和测试集的列差异，并补充测试集中的缺失列
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # 给缺失的列添加 0 值

# 确保列的顺序与训练集一致
test_df = test_df[train_df.columns]

# 特征与目标变量分开
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# 数据拆分为训练集和测试集 (80%训练，20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 用测试集进行预测
y_pred = model.predict(X_test)

# 评估模型的性能
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # 决定系数 (R^2)

# 输出结果
print(f"均方误差 (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# 绘制真实值与预测值的对比图
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # 添加理想线
plt.xlabel("实际房价")
plt.ylabel("预测房价")
plt.title("实际房价 vs 预测房价")
plt.show()

# 用训练好的模型对测试集进行预测
X_test_final = scaler.transform(test_df.drop('SalePrice', axis=1, errors='ignore'))  # 测试集没有 SalePrice 列
test_predictions = model.predict(X_test_final)

# 保存预测结果
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_predictions})
submission.to_csv('submission.csv', index=False)

print("预测结果已保存到 submission.csv")
