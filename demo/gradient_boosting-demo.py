import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from magilearn.models import GradientBoostingClassifier

# 1. 生成模拟的二分类数据
np.random.seed(42)
X = np.random.randn(1000, 2)  # 特征数据，1000个样本，2个特征
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 使用简单的线性组合生成二分类标签

# 2. 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 自定义梯度提升分类器训练
my_gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
my_gbc.fit(X_train, y_train)
my_y_pred = my_gbc.predict(X_test)
my_proba = my_gbc.predict_proba(X_test)[:, 1]  # 正类概率

# 4. Sklearn 的梯度提升分类器训练
sk_gbc = SklearnGBC(n_estimators=50, learning_rate=0.1, max_depth=3)
sk_gbc.fit(X_train, y_train)
sk_y_pred = sk_gbc.predict(X_test)
sk_proba = sk_gbc.predict_proba(X_test)[:, 1]

# 5. 评价两个模型的性能
my_acc = accuracy_score(y_test, my_y_pred)
sk_acc = accuracy_score(y_test, sk_y_pred)

my_auc = roc_auc_score(y_test, my_proba)
sk_auc = roc_auc_score(y_test, sk_proba)

print(f"Magilearn GradientBoostingClassifier 准确率: {my_acc:.4f}, AUC: {my_auc:.4f}")
print(f"Sklearn GradientBoostingClassifier 准确率: {sk_acc:.4f}, AUC: {sk_auc:.4f}")

# 6. 可视化两个模型的比较
plt.figure(figsize=(12, 6))

# 设置中文字体（适用于中文环境，非必要可以注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制决策边界
def plot_decision_boundary(model, X, y, title, subplot):
    h = 0.02  # 网格间隔
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, subplot)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", marker="o")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# 自定义 GBC 的决策边界
plot_decision_boundary(my_gbc, X_test, y_test, "Magilearn GradientBoosting", 1)

# Sklearn GBC 的决策边界
plot_decision_boundary(sk_gbc, X_test, y_test, "Sklearn GradientBoosting", 2)

plt.tight_layout()
plt.show()
