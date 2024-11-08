import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel

class SVC(BaseEstimator):
    def __init__(self, C=1.0, kernel='linear'):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        self.X_support = None

    def _kernel_function(self, X):
        """计算核函数，当前仅支持线性核"""
        if self.kernel == 'linear':
            return linear_kernel(self.X_support, X)  # 这里是使用训练后的支持向量
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    def fit(self, X, y):
        """训练支持向量机模型"""
        n_samples, n_features = X.shape

        # 将标签转换为 +1 和 -1
        y = np.where(y == 0, -1, 1)

        # 计算内积并计算支持向量机的核矩阵
        K = linear_kernel(X, X)  # 使用训练集样本之间的内积

        # 初始化拉格朗日乘子（alpha）
        alpha = np.zeros(n_samples)

        # 设置最大迭代次数
        max_iter = 1000
        epsilon = 1e-5
        for _ in range(max_iter):
            # 计算目标函数梯度
            grad = np.ones(n_samples) - y * (K.dot(alpha * y))

            # 使用梯度下降进行优化
            alpha_new = alpha + self.C * grad
            alpha_new = np.clip(alpha_new, 0, self.C)

            # 计算误差
            diff = np.linalg.norm(alpha_new - alpha)
            if diff < epsilon:
                break
            alpha = alpha_new

        # 选择支持向量
        support_vectors = alpha > 1e-4
        self.support_vectors = X[support_vectors]
        self.alpha = alpha[support_vectors]
        self.support_vector_labels = y[support_vectors]
        self.X_support = X[support_vectors]  # 正确初始化支持向量

        # 计算截距项
        self.intercept = np.mean(self.support_vector_labels - np.dot(self._kernel_function(self.X_support), self.alpha * self.support_vector_labels))

    def predict(self, X):
        """预测类别"""
        K = self._kernel_function(X)
        
        # 计算决策函数时，确保 alpha 和 support_vector_labels 的形状正确
        decision_function = np.dot(K, self.alpha * self.support_vector_labels) + self.intercept
        
        # 预测类别
        return np.sign(decision_function)

# 测试代码
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # 1. 生成模拟数据
    X, y = make_classification(
        n_samples=1000,         # 样本数量
        n_features=20,          # 特征数量
        n_classes=2,            # 分类数量（2 表示二分类）
        random_state=42
    )

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. 训练支持向量机模型
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)

    # 5. 模型预测
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    # 6. 评估模型性能
    print("训练集准确率:", accuracy_score(y_train, y_pred_train))
    print("测试集准确率:", accuracy_score(y_test, y_pred_test))
    print("\n混淆矩阵 (测试集):\n", confusion_matrix(y_test, y_pred_test))
    print("\n分类报告 (测试集):\n", classification_report(y_test, y_pred_test))
