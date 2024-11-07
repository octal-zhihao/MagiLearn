import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectFromModel
# 去掉sklearn.后的SelectFromModel效果一致
from feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化决策树分类器
model = DecisionTreeClassifier(random_state=42)

# 使用决策树进行训练
model.fit(X_train, y_train)

# 使用SelectFromModel来选择重要特征
selector = SelectFromModel(model, threshold="mean")  # 选择重要性大于平均值的特征
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 打印被选择的特征索引
print("Selected features:", selector.get_support(indices=True))

# 使用原始特征进行预测
y_pred = model.predict(X_test)
# 计算并输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on original features: {accuracy:.4f}")

# 使用选择的特征重新训练模型
model_selected = DecisionTreeClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)

# 使用训练好的模型进行预测
y_pred = model_selected.predict(X_test_selected)

# 计算并输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on selected features: {accuracy:.4f}")
