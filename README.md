# MagiLearn

24秋季学期机器学习课程作业

- 总目标：实现一个自制机器学习库

- 中期目标：具备sklearn库的基础功能，能完成一个demo

> 库的名称待定，咱们可以想一个 `xxx-learn`，确定后再更改项目名称


### 暂定项目结构如下:



```bash
MagiLearn/
│
├── datasets/                 # 数据集模块
│   ├── __init__.py           # 初始化模块
│   └── load_iris.py          # 加载Iris数据集的实现
│   └── make_classification.py # 生成模拟数据
│
├── feature_selection/        # 特征选择与降维模块
│   ├── __init__.py           # 初始化模块
│   ├── select_k_best.py      # 特征选择
│   └── pca.py                # 主成分分析
│
├── model_selection/          # 模型选择与评估模块
│   ├── __init__.py           # 初始化模块
│   ├── train_test_split.py   # 数据集划分
│   └── grid_search.py        # 网格搜索
│   └── cross_val_score.py    # 交叉验证
│
├── models/                   # 存放各种机器学习模型
│   ├── __init__.py           # 初始化模块
│   ├── logistic_regression.py # 逻辑回归模型
│   ├── linear_regression.py  # 线性回归模型
│   ├── svm.py                # 支持向量机分类器
│   ├── decision_tree.py      # 决策树模型
│   ├── random_forest.py      # 随机森林分类器
│   ├── gradient_boosting.py  # 梯度提升分类器
│   └── k_means.py            # K均值聚类模型
│
├── preprocessing/            # 数据预处理模块
│   ├── __init__.py           # 初始化模块
│   ├── scaler.py             # 数据缩放 (StandardScaler, MinMaxScaler)
│   ├── encoder.py            # 数据编码 (OneHotEncoder, LabelEncoder)
│
├── metrics/                  # 模型评估模块
│   ├── __init__.py           # 初始化模块
│   ├── accuracy.py           # 准确率
│   ├── precision_recall.py   # 精确率与召回率
│   ├── confusion_matrix.py   # 混淆矩阵
│   └── roc_auc.py            # ROC AUC评估
│
├── tests/                    # 测试代码
│   ├── test_logistic_regression.py
│   ├── test_scaler.py
│   └── ...
├── pipeline/                 # 管道模块
│   ├── __init__.py           # 初始化模块
│   ├── pipeline.py            # 实现自定义管道的功能
│   └── pipeline_utils.py      # 辅助功能或工具函数
│   ...
│   
├── __init__.py               # 顶层模块
└── README.md                 # 项目说明文件
```


### 主要模块实现介绍

1. **数据预处理 (`preprocessing`)**
   - `scaler.py`: 实现 `StandardScaler` 和 `MinMaxScaler`。
   - `encoder.py`: 实现 `OneHotEncoder` 和 `LabelEncoder`。
   
2. **模型选择与评估 (`model_selection`)**
   - `train_test_split.py`: 实现数据集划分函数。
   - `grid_search.py`: 实现网格搜索算法 (`GridSearchCV`)。
   - `cross_val_score.py`: 实现交叉验证函数。

3. **特征选择与降维 (`feature_selection` 和 `decomposition`)**
   - `select_k_best.py`: 实现 `SelectKBest` 算法，选择最佳特征。
   - `pca.py`: 实现 `PCA` 降维算法。

4. **分类模型 (`models`)**
   - `logistic_regression.py`: 实现逻辑回归模型。
   - `svm.py`: 实现 `SVM` 分类模型。
   - `decision_tree.py`: 实现决策树分类器 (`DecisionTreeClassifier`)。
   - `random_forest.py`: 实现随机森林分类器 (`RandomForestClassifier`)。
   - `gradient_boosting.py`: 实现梯度提升分类器 (`GradientBoostingClassifier`)。

5. **回归模型 (`models`)**
   - `linear_regression.py`: 实现线性回归 (`LinearRegression`)。
   - `ridge.py` 和 `lasso.py`: 分别实现岭回归 (`Ridge`) 和 `Lasso` 回归模型。

6. **聚类算法 (`models`)**
   - `k_means.py`: 实现 K 均值聚类算法。
   - `dbscan.py`: 实现基于密度的 DBSCAN 聚类算法。

7. **模型评估 (`metrics`)**
   - `accuracy.py`: 实现准确率度量 (`accuracy_score`)。
   - `precision_recall.py`: 实现精确率与召回率 (`precision_score`, `recall_score`)。
   - `confusion_matrix.py`: 实现混淆矩阵 (`confusion_matrix`)。
   - `roc_auc.py`: 实现 `ROC AUC` 分数的评估 (`roc_auc_score`)。

8. **管道 (`pipeline`)**
   - `pipeline.py`: 实现 `Pipeline` 类，用于将数据预处理和模型训练过程串联起来。

### 项目框架说明
- **层次结构清晰**：每个模块功能单一，例如模型存放在 `models` 文件夹，数据预处理在 `preprocessing`，评估指标在 `metrics` 中等。
- **方便扩展**：在已有结构基础上，后续可以轻松增加新的模型、算法和功能模块。
- **易于测试**：将每个功能模块拆分成单独的测试文件，可以在 `tests/` 文件夹中针对不同模型进行单元测试，确保代码正确性。


> *based on gpt's suggestion*
