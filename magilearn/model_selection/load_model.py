import joblib

def load_model(filename):
    """
    从文件加载已保存的模型。

    参数：
        filename : str
            模型文件路径和名称
    
    返回：
        model : 已加载的模型对象
    """
    try:
        model = joblib.load(filename)
        print(f"模型已从 {filename} 加载")
        return model
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None
