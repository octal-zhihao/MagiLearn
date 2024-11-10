import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 实现RFE后更改下面一行
from sklearn.feature_selection import RFE
# from magilearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=200)

# 使用RFE进行特征选择，这里我们选择最重要的2个特征
selector = RFE(estimator=model, n_features_to_select=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 打印被选择的特征索引
print("Selected features:", selector.get_support(indices=True))

# 使用逻辑回归进行训练
model.fit(X_train_selected, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_selected)

# 计算并输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on selected features: {accuracy:.4f}")
