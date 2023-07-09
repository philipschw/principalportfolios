import numpy as np


def expsmoother(alpha: float, data: np.ndarray, beta: float):
    """
    Perform double exponential smoothing (Holt Linear).
    """
    smoothed_data = np.zeros_like(data)
    b = np.zeros_like(data)

    smoothed_data[0] = data[0]
    b[0] = data[1] - data[0]

    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * (smoothed_data[i-1] + b[i-1])
        b[i] = beta * (smoothed_data[i] - smoothed_data[i-1]) + (1 - beta) * b[i-1]

    return smoothed_data
