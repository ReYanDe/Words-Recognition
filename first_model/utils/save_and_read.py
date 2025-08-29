import numpy as np
import struct

# созднание рандомных весов для нейросети
def initialize_weights(path, input_size, hidden_size, hidden2_size, output_size):
    # вход -> скрыйтый слой
    w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
    b1 = np.zeros((1, hidden_size))

    # скрытый слой 1 -> скрытый слой 2
    w2 = np.random.randn(hidden_size, hidden2_size) * np.sqrt(1.0 / hidden_size)
    b2 = np.zeros((1, hidden2_size))

    # скрытый слой 2 -> выходной слой
    w3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1.0 / hidden2_size)
    b3 = np.zeros((1, output_size))

    """Прим. Если тут во время указывания переменной вписывать просто сами переменные без знака = (типо w1, b1, w2, b2) то тогда numpy не создаст ключи и их значения и просто кинет все переменные как есть"""
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    print(f'Данные успешно созданы и сохранены в файл {path}.npz')

# initialize_weights("./db/weights", 784, 256, 128, 26)


# сохранение всех весов и смещений
def save_weights(path, w1, b1, w2, b2, w3, b3):
    np.savez(path, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


# загрузка весов и смещений из базы данных
def load_weights(path):
    data = np.load(path)
    w1, b1, w2, b2, w3, b3 = data['w1'], data['b1'], data['w2'], data['b2'], data['w3'], data['b3']
    return w1, b1, w2, b2, w3, b3


"""Примечание База данных EMNIST использует файлы с расширением .idx3-ubyte для ИЗОБРАЖЕНИЙ а для меток тоесть 
правильных ответов используется файл с расширением .idx1-ubyte
"""


# функция для загрузки изображения
def load_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return data
    

# функция для загрузки меток (правильных ответов)
def load_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))   # читаем заголовок
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
# images = load_images("./archive/emnist_source_files/emnist-letters-train-images-idx3-ubyte")
# labels = load_labels("./archive/emnist_source_files/emnist-letters-train-labels-idx1-ubyte")

# print(images[0])
# print(labels[0])