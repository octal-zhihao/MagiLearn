import joblib

def save_model(model, filename):
    """
    保存训练好的模型到文件。

    参数：
        model : 训练好的模型对象
        filename : str
            模型保存的文件路径和名称
    """
    try:
        joblib.dump(model, filename)
        print(f"模型已保存至 {filename}")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")
