import sys
sys.path.insert(1, "./Module")
from Generalization import Generalization
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matrix = np.array([
        [29, 12, 579, 1],
        [27, 10, 950, 1],
        [24, 10, 624, 2],
        [37, 11, 887, 3],
        [30, 8, 803, 3]
    ])
    columns = ["Стоимость пакета интернета, руб", "Скорость интернета, Мбит/с", "Поддержка, 1-5", "Стоимость звонков, руб/мин"]
    gen = Generalization(matrix)
    min_array, max_array = gen.get_min_max_arrays()
    normalized_matrix = gen.get_normalized_matrix()
    additive_convolution_1 = gen.get_additive_convolution(normalized_matrix)
    weights = [0.2, 0.4, 0.1, 0.3]
    additive_convolution_2 = gen.get_additive_convolution(normalized_matrix, method="specified", weights=weights)
    print("Матрица провайдеров:")
    print(pd.DataFrame(matrix, index=[f"A{i}" for i in range(1, 6)], columns=columns))
    print("Минимальные значения:")
    print(min_array)
    print("Максимальные значения:")
    print(max_array)
    print("Нормализованная матрица:")
    print(pd.DataFrame(normalized_matrix, index=[f"A{i}" for i in range(1, 6)], columns=columns))
    print()
    print("Аддитивная свертка (критерии равнозначны)")
    print(pd.DataFrame(additive_convolution_1, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))
    print("Аддитивная свертка (веса критериев заданы)")
    print(pd.DataFrame(additive_convolution_2, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))