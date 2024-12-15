# 模型评估模块 (metrics)

## 1.功能概览

- **分类评估指标**：准确率、混淆矩阵、精确率、召回率和 ROC AUC。

以下是每个功能的使用示例。

## 2. 分类评估指标

### 2.1 准确率 (Accuracy Score)

**准确率**是正确预测样本的比例，用于衡量分类模型的整体表现。

#### 示例代码
```python
from magilearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0]  # 真实标签
y_pred = [1, 0, 1, 0, 0]  # 预测标签

accuracy = accuracy_score(y_true, y_pred)
print("准确率:", accuracy)
```

准确率: 0.8

### 2.2 混淆矩阵 (Confusion Matrix)

**混淆矩阵**用于展示分类模型的分类效果，矩阵的每一行表示实际类，列表示预测类。适用于多分类问题。

#### 示例代码
```python
from magilearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]

matrix = confusion_matrix(y_true, y_pred)
print("混淆矩阵:\n", matrix)
```

混淆矩阵:
 [[2 0]
 [1 2]]

### 2.3 精确率和召回率 (Precision & Recall)

- **精确率**：正确预测的正类样本占所有预测为正的样本的比例。
- **召回率**：正确预测的正类样本占实际正类样本的比例。

#### 示例代码
```python
from magilearn.metrics import precision_score, recall_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print("精确率:", precision)
print("召回率:", recall)
```

精确率: 1.0
召回率: 0.6666666666666666

### 2.4 ROC AUC (Receiver Operating Characteristic - Area Under Curve)

**ROC AUC** 用于评估分类模型的判别能力，值越接近 1 表明模型的分类效果越好。

#### 示例代码
```python
from magilearn.metrics import roc_auc_score

y_true = [1, 0, 1, 1, 0]
y_scores = [0.8, 0.4, 0.9, 0.6, 0.3]  # 预测分数或概率

auc = roc_auc_score(y_true, y_scores)
print("ROC AUC:", auc)
```

ROC AUC: 1.0
