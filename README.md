# MagiLearn

MagiLearn 是一个轻量级的自定义机器学习库，旨在提供类似 `scikit-learn` 的功能，支持特征选择、模型选择、数据预处理、分类、回归、聚类等模块。通过模块化设计，MagiLearn 便于扩展并易于集成在机器学习项目中。

## Document

项目文档地址：https://magilearn-tutorial.readthedocs.io/

## 安装
MagiLearn 使用 Python 3.x 开发，推荐使用虚拟环境进行安装。

```bash
# 克隆项目
git clone https://github.com/YourGitHubUsername/MagiLearn.git
cd MagiLearn
```
```bash
# 安装依赖项
pip install -r requirements.txt
```


## 快速开始
以下是一个使用 MagiLearn 进行简单逻辑回归的示例：

```python
from magilearn.datasets import load_iris, train_test_split
from magilearn.models import LogisticRegression

# 加载数据集
X, y = load_iris(return_X_y=True)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并评估
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
```

## 主要模块API文档
- [数据预处理模块 (`preprocessing`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/preprocessing.md)
- [模型选择与评估模块 (`model_selection`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/model_selection.md)
- [特征选择与降维模块 (`feature_selection`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/feature_selection.md)
- [分类与回归模型 (`models`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/models.md)
- [模型评估模块 (`metrics`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/metrics.md)
- [管道模块 (`pipeline`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/pipline.md)
- [数据集模块 (`datasets`)](https://github.com/octal-zhihao/MagiLearn/blob/main/doc/datasets.md)



## 现阶段项目结构

```bash
magilearn/
│
├── datasets/                 # 数据集模块
│   ├── __init__.py           # 初始化模块
│   ├── train_test_split.py   # 数据集划分
│   ├── load_iris.py          # 加载Iris数据集
│   └── make_classification.py # 生成模拟数据
│
├── feature_selection/        # 特征选择与降维模块
│   ├── __init__.py           # 初始化模块
│   ├── select_k_best.py      # 特征选择
│   ├── REF.py                # 递归特征消除 (RFE)
│   ├── select_from_model.py  # 基于模型选择特征
│   └── pca.py                # 主成分分析
│
├── model_selection/          # 模型选择与评估模块
│   ├── __init__.py           # 初始化模块
│   ├── grid_search.py        # 网格搜索
│   ├── save_model.py         # 保存模型
│   ├── load_model.py         # 加载模型
│   └── cross_val_score.py    # 交叉验证
│
├── models/                   # 存放各种机器学习模型
│   ├── __init__.py           # 初始化模块
│   ├── logistic_regression.py # 逻辑回归模型
│   ├── linear_regression.py  # 线性回归模型
│   ├── decision_tree.py      # 决策树模型
│   ├── random_forest.py      # 随机森林分类器
│   ├── gradient_boosting.py  # 梯度提升分类器
│   └── k_means.py            # K均值聚类模型
│
├── pipeline/                 # 管道模块
│   ├── __init__.py           # 初始化模块
│   └── pipeline.py            # 自定义Pipeline实现
│
├── preprocessing/            # 数据预处理模块
│   ├── __init__.py           # 初始化模块
│   ├── scaler.py             # 数据缩放 (StandardScaler, MinMaxScaler, RobustScaler)
│   ├── encoder.py            # 数据编码 (OneHotEncoder, LabelEncoder, LabelBinarizer)
│   └── normalizer.py         # 归一化 Normalizer
│
├── metrics/                  # 模型评估模块
│   ├── __init__.py           # 初始化模块
│   ├── accuracy.py           # 准确率
│   ├── precision_recall.py   # 精确率与召回率
│   ├── confusion_matrix.py   # 混淆矩阵
│   └── roc_auc.py            # ROC AUC评估
│
│   ...
│   
├── __init__.py               # 顶层模块
└── README.md                 # 项目说明文件
```


### 主要模块实现介绍

1. **数据预处理 (`preprocessing`)**
   - `scaler.py`: 实现 `StandardScaler` 和 `MinMaxScaler`。
   - `encoder.py`: 实现 `OneHotEncoder` 和 `LabelEncoder`。

2. **数据集 (`datasets`)**
   - `load_iris.py`: 实现加载Iris数据集
   - `make_classification.py` 生成模拟分类数据
   - `train_test_split.py`: 实现数据集划分函数
   
3. **模型选择与评估 (`model_selection`)**
   - `grid_search.py`: 实现网格搜索算法 (`GridSearchCV`)。
   - `cross_val_score.py`: 实现交叉验证函数。
   - `save_model.py`: 实现模型参数的保存。
   - `load_model.py`: 实现模型参数的载入。

4. **特征选择与降维 (`feature_selection` 和 `decomposition`)**
   - `select_k_best.py`: 实现 `SelectKBest` 算法，选择最佳特征。
   - `pca.py`: 实现 `PCA` 降维算法。
   - `REF.py`: 实现 `REF` 递归特征消除。

5. **分类模型 (`models`)**
   - `logistic_regression.py`: 实现逻辑回归模型。
   - `decision_tree.py`: 实现决策树分类器 (`DecisionTreeClassifier`)。
   - `random_forest.py`: 实现随机森林分类器 (`RandomForestClassifier`)。
   - `gradient_boosting.py`: 实现梯度提升分类器 (`GradientBoostingClassifier`)。

6. **回归模型 (`models`)**
   - `linear_regression.py`: 实现线性回归 (`LinearRegression`)。
   - `ridge.py` 和 `lasso.py`: 分别实现岭回归 (`Ridge`) 和 `Lasso` 回归模型。

7. **聚类算法 (`models`)**
   - `k_means.py`: 实现 K 均值聚类算法。
   - `dbscan.py`: 实现基于密度的 DBSCAN 聚类算法。

8. **模型评估 (`metrics`)**
   - `accuracy.py`: 实现准确率度量 (`accuracy_score`)。
   - `precision_recall.py`: 实现精确率与召回率 (`precision_score`, `recall_score`)。
   - `confusion_matrix.py`: 实现混淆矩阵 (`confusion_matrix`)。
   - `roc_auc.py`: 实现 `ROC AUC` 分数的评估 (`roc_auc_score`)。

9. **管道 (`pipeline`)**
   - `pipeline.py`: 实现 `Pipeline` 类，用于将数据预处理和模型训练过程串联起来。



## 贡献指南
欢迎对 MagiLearn 提出反馈和贡献代码！如果你有好的想法或发现问题，请通过提交 Issue 或 Pull Request 参与我们的项目。

### 如何贡献
1. Fork 本项目并创建新分支
2. 进行修改和改进
3. 提交 Pull Request

## 许可证 | License

MagiLearn 遵循 [MIT License](https://github.com/octal-zhihao/MagiLearn/blob/main/LICENSE) 开源协议。
