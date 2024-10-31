from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.preprocessing import StandardScaler
from preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from metrics import calculate_accuracy as accuracy_score, calculate_confusion_matrix as confusion_matrix, calculate_precision, calculate_recall
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np
from metrics import calculate_roc_auc

# 1. 生成模拟数据集
X, y = make_classification(
    n_samples=1000,         # 样本数
    n_features=20,          # 特征数
    n_classes=2,            # 类别数
    random_state=42         # 随机种子
)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 数据预处理 + 模型构建使用Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # 数据标准化
    ('classifier', LogisticRegression())       # 逻辑回归模型
])

# 4. 参数网格搜索 (GridSearchCV) 设置超参数组合
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],      # 正则化强度
    'classifier__solver': ['liblinear', 'lbfgs']  # 求解器选择
}

# 5. 交叉验证 (5折) + 网格搜索寻找最佳参数组合
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# 6. 使用最优参数训练模型
best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# 7.获取训练集和测试集的预测分数（概率）用于计算 ROC AUC
y_train_scores = best_model.predict_proba(X_train)[:, 1]  # 获取正类概率
y_test_scores = best_model.predict_proba(X_test)[:, 1]    # 获取正类概率
# 计算并输出 ROC AUC 分数
train_auc = calculate_roc_auc(y_train, y_train_scores)
test_auc = calculate_roc_auc(y_test, y_test_scores)

# 8. 评估模型性能
print("\n训练集准确率:", accuracy_score(y_train, y_pred_train))
print("测试集准确率:", accuracy_score(y_test, y_pred_test))
print("\n训练集精确确率:", calculate_precision(y_train, y_pred_train))
print("测试集精确率:", calculate_precision(y_test, y_pred_test))
print("\n训练集召回率:", calculate_recall(y_train, y_pred_train))
print("测试集召回率:", calculate_recall(y_test, y_pred_test))
print("\n训练集 ROC AUC:", train_auc)
print("测试集 ROC AUC:", test_auc)
print("\n混淆矩阵 (测试集):\n", confusion_matrix(y_test, y_pred_test))
print("\n分类报告 (测试集):\n", classification_report(y_test, y_pred_test))

# 9. 使用交叉验证进一步验证模型稳定性
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print("\n交叉验证准确率均值:", np.mean(cv_scores))
print("交叉验证准确率标准差:", np.std(cv_scores))
