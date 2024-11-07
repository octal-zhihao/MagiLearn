from datasets import make_classification
from preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from models.logistic_regression import LogisticRegression
from pipeline import Pipeline

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 创建管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# 拟合模型
pipeline.fit(X, y)

# 预测
y_pred = pipeline.predict(X)

# 计算准确率
accuracy = pipeline.score(X, y)
print(f"模型准确率: {accuracy:.2f}")
