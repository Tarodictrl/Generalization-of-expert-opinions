import sys
sys.path.insert(1, "./Module")
from Generalization import Generalization
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matrix = np.array([
        [140, 70, 2, 1],
        [200, 90, 1, 1],
        [180, 60, 1, 3],
        [120, 90, 5, 1],
        [170, 80, 2, 5]
    ])
    weights = [0.2, 0.4, 0.1, 0.3]
    columns = ["Стоимость пакета интернета, руб", "Скорость интернета, Мбит/с", 
               "Поддержка, 1-5", "Стоимость звонков, руб/мин"]
    gen = Generalization(matrix)
    min_array, max_array = gen.get_min_max_arrays()
    normalized_matrix = gen.get_normalized_matrix()
    additive_convolution_1 = gen.get_additive_convolution(normalized_matrix)
    additive_convolution_2 = gen.get_additive_convolution(normalized_matrix, 
                                                          method="specified", 
                                                          weights=weights)
    additive_convolution_1_weights = gen.get_criterion_weight(additive_convolution_1)
    additive_convolution_2_weights = gen.get_criterion_weight(additive_convolution_2)
     
    multiplicative_convolution_1 = gen.get_multiplicative_convolution(normalized_matrix)
    multiplicative_convolution_2 = gen.get_multiplicative_convolution(normalized_matrix, 
                                                                      method="specified", 
                                                                      weights=weights)
    multiplicative_convolution_1_weights = gen.get_criterion_weight(multiplicative_convolution_1)
    multiplicative_convolution_2_weights = gen.get_criterion_weight(multiplicative_convolution_2)
    
    perfect_point_1 = gen.get_perfect_point(normalized_matrix)
    perfect_point_2 = gen.get_perfect_point(normalized_matrix, 
                                            method="specified", 
                                            weights=weights)
    perfect_point_1_weights = gen.get_criterion_weight(perfect_point_1)
    perfect_point_2_weights = gen.get_criterion_weight(perfect_point_2)
    
    print("Матрица провайдеров:")
    print(pd.DataFrame(matrix, index=[f"A{i}" for i in range(1, 6)], columns=columns))
    print("Веса:")
    print(weights)
    print("Минимальные значения:")
    print(min_array)
    print("Максимальные значения:")
    print(max_array)
    print("Нормализованная матрица:")
    print(pd.DataFrame(normalized_matrix, 
                       index=[f"A{i}" for i in range(1, 6)], 
                       columns=columns))
    print()
    print("Аддитивная свертка (критерии равнозначны)")
    print(pd.DataFrame(zip(additive_convolution_1, additive_convolution_1_weights[1]), 
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(additive_convolution_1_weights[0])
    print()
    print("Аддитивная свертка (веса критериев заданы)")
    print(pd.DataFrame(zip(additive_convolution_2, additive_convolution_2_weights[1]),
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(additive_convolution_2_weights[0])
    print()
    print("Мультипликативная свертка частных критериев (критерии равнозначны)")
    print(pd.DataFrame(zip(multiplicative_convolution_1, multiplicative_convolution_1_weights[1]),
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(multiplicative_convolution_1_weights[0])
    print()
    print("Мультипликативная свертка частных критериев (веса критериев заданы)")
    print(pd.DataFrame(zip(multiplicative_convolution_2, multiplicative_convolution_2_weights[1]),
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(multiplicative_convolution_2_weights[0])
    print()
    print("Метод идеальной точки (критерии равнозначны)")
    print(pd.DataFrame(zip(perfect_point_1, perfect_point_1_weights[1]),
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(perfect_point_1_weights[0])
    print()
    print("Метод идеальной точки (веса критериев заданы)")
    print(pd.DataFrame(zip(perfect_point_2, perfect_point_2_weights[1]),
                       index=[f"A{i}" for i in range(1, 6)], columns=["Свертка", "Веса критерия"]))
    print("Наиболее предпочтительней является:")
    print(perfect_point_2_weights[0])
    