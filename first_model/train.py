import numpy as np 
import math

from utils.save_and_read import *
from utils.math_utils import *
from utils.data_loader import *
from model import *


# загрузка данных из базы данных EMNIST
images = load_images("./archive/emnist_source_files/emnist-letters-train-images-idx3-ubyte")
labels = load_labels("./archive/emnist_source_files/emnist-letters-train-labels-idx1-ubyte")

# переменные для нейронов изображения и правильного ответа
# img_neuron, true_otv = get_random_image(images, labels)


# img_neuron = img_neuron.flatten()
    
"""
hidden_weights_1 - весы первого скрытого слоя; bias_1 - смещения первого скрытого слоя; 
hidden_weights_2 - весы второго скрытого слоя; bias_2 - смещения второго скрытого слоя; 
output_weights_3 - весы выходного слоя; bias_3 - смещения выходного слоя; 
"""

hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias = load_weights('db/weights.npz')

learning_step = 0.001
EPOCH = 20


for epoch in range(EPOCH):
    # переменные для просчета ошибки нейросети
    total_loss = 0
    correct_predictions = 0

    for i in range(len(images)):

        # нормализуем входные данные (от 0 до 1)
        img_neuron = (images[i].astype(np.float32) / 255.0).flatten()
        true_otv = labels[i] - 1


        # feedforward
        z1, a_hidden_1, z2, a_hidden_2, z3, res_softmax, predicted_number = feedforward(img_neuron, hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias)

        # backpropagation
        hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias, loss = backpropagation(z1, a_hidden_1, z2, a_hidden_2, res_softmax, img_neuron, true_otv, hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias, learning_step)

        total_loss += loss
        correct_predictions += int(predicted_number == true_otv)


    avg_loss = total_loss / len(images)
    accuracy = correct_predictions / len(images) * 100
    print(f"Эпоха {epoch + 1}/{EPOCH} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

save_weights('db/weights.npz', hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias)