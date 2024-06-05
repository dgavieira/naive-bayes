import numpy as np


def pearsonr_custom(x, y):
    """
    Calcula o coeficiente de Pearson entre dois arrays.
    
    Args:
    x (numpy.ndarray): Primeiro array.
    y (numpy.ndarray): Segundo array.
    
    Returns:
    float: Coeficiente de Pearson.
    """
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x), np.std(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    
    if std_x == 0 or std_y == 0:
        return 0
    
    return covariance / (std_x * std_y)