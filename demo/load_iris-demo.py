from magilearn.datasets.load_iris import load_iris

# 加载数据集
iris = load_iris()

# 查看数据集信息
print("特征名称:", iris['feature_names'])
print("数据形状:", iris['data'].shape)
print("目标名称:", iris['target_names'])
print("描述信息:", iris['DESCR'])
print("数据示例:")
print(iris['data'][:5])  # 打印前5个样本
print("目标示例:")
print(iris['target'][:5])  # 打印前5个标签
