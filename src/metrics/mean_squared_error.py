import numpy as np

def mean_squared_error_custom(y_true, y_pred):
    """
    Calcula o erro quadrático médio entre os valores verdadeiros e previstos.
    
    Args:
    y_true (numpy.ndarray): Valores verdadeiros.
    y_pred (numpy.ndarray): Valores previstos.
    
    Returns:
    float: Erro quadrático médio.
    """
    return np.mean((y_true - y_pred) ** 2)