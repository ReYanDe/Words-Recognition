import numpy as np
import time 



from model import *
from utils.save_and_read import *
from utils.math_utils import *
from utils.data_loader import *
from model import *


# загрузка данных из базы данных EMNIST
images = load_images("./archive/emnist_source_files/emnist-letters-test-images-idx3-ubyte")
labels = load_labels("./archive/emnist_source_files/emnist-letters-test-labels-idx1-ubyte")

# переменные для нейронов изображения и правильного ответа
img_neuron, true_otv = get_random_image(images, labels)


img_neuron = img_neuron.flatten()
    
"""
hidden_weights_1 - весы первого скрытого слоя; bias_1 - смещения первого скрытого слоя; 
hidden_weights_2 - весы второго скрытого слоя; bias_2 - смещения второго скрытого слоя; 
output_weights_3 - весы выходного слоя; bias_3 - смещения выходного слоя; 
"""

hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias = load_weights('db/weights.npz')

w = 'abcdefghijklmnopqrstuvwxyz'.title()
words = list(w)


for i in range(10):

    img_neuron, true_otv = get_random_image(images, labels)
    img_neuron = img_neuron.flatten()

    # feedforward
    z1, a_hidden_1, z2, a_hidden_2, z3, res_softmax, predicted_number = feedforward(img_neuron, hidden_weights_1, bias_1, hidden_weights_2, bias_2, output_weights, output_bias)

    print(f'Предугадано: {words[predicted_number]}, Правильно: {words[true_otv - 1]}')
    time.sleep(1)