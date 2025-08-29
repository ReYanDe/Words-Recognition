import numpy as np


# нормализует изображение а также делает рандомный выбор изображения
def get_random_image(img, label):
    ind = np.random.randint(0, len(img))
    images = np.round(img.astype(np.float32) / 255.0, 4)
    return images[ind], label[ind] 