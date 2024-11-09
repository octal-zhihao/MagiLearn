from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
from magilearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from magilearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report
import numpy as np

from magilearn.metrics import calculate_roc_auc

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


# 6. 获取预测分数（概率）用于 ROC AUC 计算
y_train_scores = model.predict_proba(X_train_selected)[:, 1]  # 获取正类概率
y_test_scores = model.predict_proba(X_test_selected)[:, 1]    # 获取正类概率

# 7. 评估模型性能
print("测试集准确率:", accuracy_score(y_test, y_pred))
print("测试集精确率:", precision_score(y_test, y_pred))
print("测试集召回率:", recall_score(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 计算并输出 ROC AUC 分数
train_auc = calculate_roc_auc(y_train, y_train_scores)
test_auc = calculate_roc_auc(y_test, y_test_scores)

print("\n训练集 ROC AUC:", train_auc)
print("测试集 ROC AUC:", test_auc)


# 8. 特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n特征重要性:")
for f in range(X_train_selected.shape[1]):
    print(f"特征 {indices[f]}: {importances[indices[f]]:.4f}")
