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
    weights = [0.2, 0.4, 0.1, 0.3]
    columns = ["Стоимость пакета интернета, руб", "Скорость интернета, Мбит/с", "Поддержка, 1-5", "Стоимость звонков, руб/мин"]
    gen = Generalization(matrix)
    min_array, max_array = gen.get_min_max_arrays()
    normalized_matrix = gen.get_normalized_matrix()
    additive_convolution_1 = gen.get_additive_convolution(normalized_matrix)
    additive_convolution_2 = gen.get_additive_convolution(normalized_matrix, method="specified", weights=weights)
    multiplicative_convolution_1 = gen.get_multiplicative_convolution(normalized_matrix)
    multiplicative_convolution_2 = gen.get_multiplicative_convolution(normalized_matrix, method="specified", weights=weights)
    perfect_point_1 = gen.get_perfect_point(normalized_matrix)
    perfect_point_2 = gen.get_perfect_point(normalized_matrix, method="specified", weights=weights)
    print("Матрица провайдеров:")
    print(pd.DataFrame(matrix, index=[f"A{i}" for i in range(1, 6)], columns=columns))
    print("Веса:")
    print(weights)
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
    print("Мультипликативная свертка (критерии равнозначны)")
    print(pd.DataFrame(multiplicative_convolution_1, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))
    print("Мультипликативная свертка (веса критериев заданы)")
    print(pd.DataFrame(multiplicative_convolution_2, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))
    print("Метод идеальной точки (критерии равнозначны)")
    print(pd.DataFrame(perfect_point_1, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))
    print("Метод идеальной точки (веса критериев заданы)")
    print(pd.DataFrame(perfect_point_2, index=[f"A{i}" for i in range(1, 6)], columns=["Свертка"]))