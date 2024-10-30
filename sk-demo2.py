from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. 生成模拟数据集
X, y = make_classification(
    n_samples=1000,         # 样本数
    n_features=20,          # 特征数
    n_informative=10,       # 有信息的特征数
    n_redundant=5,          # 冗余特征数
    random_state=42         # 随机种子
)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 数据预处理 - 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 特征选择 - 使用递归特征消除 (RFE)
model = RandomForestClassifier(random_state=42)
selector = RFE(model, n_features_to_select=10)  # 选择10个最重要的特征
selector.fit(X_train_scaled, y_train)

# 获取选择的特征
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 5. 模型训练与评估
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# 6. 评估模型性能
print("测试集准确率:", accuracy_score(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 7. 特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n特征重要性:")
for f in range(X_train_selected.shape[1]):
    print(f"特征 {indices[f]}: {importances[indices[f]]:.4f}")
