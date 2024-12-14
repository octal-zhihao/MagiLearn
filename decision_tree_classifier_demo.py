import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from magilearn.models.decision_tree_classifier import DecisionTreeClassifier

# 1. 加载鸢尾花数据集
def load_and_prepare_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 只保留两类数据 (Setosa 和 Versicolor)，将其简化为二分类
    binary_mask = (y == 0) | (y == 1)
    X, y = X[binary_mask], y[binary_mask]
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    return X, y

# 2. 数据划分
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 模型训练与预测
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # 使用自定义决策树分类器
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=4)
    clf.fit(X_train, y_train)
    
    # 预测结果
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")
    
    return clf, y_pred

# 4. 可视化结果
def visualize_results(X_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制真实值和预测值
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolor='black',
                s=100, label='真实值', alpha=0.7, facecolors='none')
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', s=50,
                label='预测值', linewidths=2, alpha=0.7, color='red')
    # 添加图例
    plt.legend(loc='upper right')
    plt.title('决策树分类器预测结果')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 主程序
if __name__ == "__main__":    
    # 加载和准备数据
    X, y = load_and_prepare_data()
    
    # 只取前两个特征方便可视化
    X = X[:, :2]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 训练模型并预测
    clf, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # 可视化预测结果
    visualize_results(X_test, y_test, y_pred)


# 创建数据集
# X = np.array([[1], [2], [3], [4], [5]])
# y = np.array([0, 0, 1, 1, 1])

# # 训练决策树
# clf = DecisionTreeClassifier(max_depth=2)
# clf.fit(X, y)

# # 预测
# predictions = clf.predict(X)
# print("Predictions:", predictions)

# # 验证结果
# assert (predictions == y).all(), "预测结果与真实标签不符！"
