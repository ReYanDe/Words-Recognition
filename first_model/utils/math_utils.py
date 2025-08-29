import math
import numpy as np

def scalar(vec1, vec2):
    res = 0.0
    for i in range(len(vec1)):
        res += vec1[i] * vec2[i]
    return res

def scalar_output_z(a_hidden: list[float], output_weight: list[float]):
    z = 0.0
    for i in range(len(a_hidden)):
        z += a_hidden[i] * output_weight[i]
    
    return z


def softmax(z):
    z = np.array(z)             
    shift_z = z - np.max(z)    
    exp_scores = np.exp(shift_z)
    return exp_scores / np.sum(exp_scores)

"""Простое решение проблемы с фиксированным сжатием информации насильно (не эффективно так как веса очень грубо сжаты )"""

# def sigmoid(x):
#     x = np.clip(x, -500, 500)  # ограничиваем диапазон
#     return 1 / (1 + np.exp(-x))

"""Рекомендованная формула сжатия информации """

def sigmoid(x):
    # для положительных и отрицательных значений разные формулы (чтобы exp не взрывался)
    out = np.empty_like(x, dtype=np.float64)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    # x >= 0: 1 / (1 + exp(-x))
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    # x < 0: exp(x) / (1 + exp(x)) — здесь exp(x) не взорвется, так как x отрицательный
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)
    
    return out

# def sigmoid_derivative(x):
#     s = sigmoid(x)
#     return s * (1 - s)