import numpy as np 
import math

# from utils.save_and_read import *
from utils.math_utils import *
# from utils.data_loader import *


# # загрузка данных из базы данных EMNIST
# images = load_images("./archive/emnist_source_files/emnist-letters-train-images-idx3-ubyte")
# labels = load_labels("./archive/emnist_source_files/emnist-letters-train-labels-idx1-ubyte")

# # переменные для нейронов изображения и правильного ответа
# img_neuron, true_otv = get_random_image(images, labels)


# img_neuron = img_neuron.flatten()
    
# """
# hidden_weights_1 - весы первого скрытого слоя; bias_1 - смещения первого скрытого слоя; 
# hidden_weights_2 - весы второго скрытого слоя; bias_2 - смещения второго скрытого слоя; 
# output_weights_3 - весы выходного слоя; bias_3 - смещения выходного слоя; 
# """

# hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias = load_weights('db/weights.npz')


"""----------------------------------------------------------------------------FEEDFORWARD-------------------------------------------------------------------------------------------"""

def feedforward(img_neuron, hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias):


    """первый скрытый слой (784 -> 256)"""
    # dot - делает вычисление скалярное  произведение векторов (матричное умножение) 
    z1 = np.dot(img_neuron, hidden_weights_1) + bias_1
    a_hidden_1 = sigmoid(z1)

    """Второй скрытый слой (256 -> 128)"""
    z2 = np.dot(a_hidden_1, hidden_weights_2) + bias_2
    a_hidden_2 = sigmoid(z2)

    """Выходной слой (128 -> 26)"""
    z3 = np.dot(a_hidden_2, output_weights) + output_bias
    res_softmax = softmax(z3).ravel()

    predicted_number = np.argmax(res_softmax)

    # print("Softmax output:", res_softmax)
    # print("Predicted class:", predicted_number)
    # print("True class:", true_otv)

    return z1, a_hidden_1, z2, a_hidden_2, z3, res_softmax, predicted_number


"""-----------------------------------------------------------BACKPROPAGATION---------------------------------------------------------------------------------"""

def backpropagation(z1, a_hidden_1, z2, a_hidden_2, res_softmax, img_neuron, true_otv, hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias, learning_step):

    # Ошибка на выходе (softamx + crossentropy)
    delta3 = res_softmax.ravel().copy()
    delta3[true_otv] -= 1


    # delta - обзначение в котором хранится значения того насколько ошиблись каждый из нейронов N-ного слоя нейросети при вычислении


    # градиенты для весов и смещений выходного слоя
    dw3 = np.outer(a_hidden_2, delta3) # (128, 26)
    db3 = delta3

    # Ошибка на втором скрытом слое 
    delta2 = (np.dot(output_weights, delta3) * (a_hidden_2 * (1 - a_hidden_2))).ravel() # (128)

    # Градиенты для весов и смещений второго скрытого слоя 
    dw2 = np.outer(a_hidden_1, delta2)
    db2 = delta2

    # ошибка на первом скрытом слое
    delta1 = (np.dot(hidden_weights_2, delta2) * (a_hidden_1 * (1 - a_hidden_1))).ravel() # (256)

    dw1 = np.outer(img_neuron, delta1) # (784 - 256)
    db1 = delta1

    # обновление весов
    hidden_weights_1 -= learning_step * dw1
    bias_1 -= learning_step * db1

    hidden_weights_2 -= learning_step * dw2
    bias_2 -= learning_step * db2

    output_weights -= learning_step * dw3
    output_bias -= learning_step * db3

    if res_softmax[true_otv] <= 0:
        res_softmax[true_otv] = 1e-9

    loss = -math.log(res_softmax[true_otv])

    return hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias, loss