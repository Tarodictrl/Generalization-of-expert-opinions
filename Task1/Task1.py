import sys
sys.path.insert(1, "./Module")
from Generalization import Generalization
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matrix = np.array([
        [10, 1, 7, 6, 9, 5, 8, 4, 3, 2],
        [8, 4, 2, 5, 1, 9, 7, 10, 6, 3],
        [9, 8, 1, 2, 10, 7, 5, 6, 3, 4],
        [7, 2, 6, 1, 9, 4, 8, 5, 3, 10],
        [6, 9, 3, 7, 10, 1, 8, 4, 2, 5]
    ])
    gen = Generalization(matrix)
    print("Матрица оценок экспертов:")
    print(pd.DataFrame(matrix, index=[f"Эксперт {i}" for i in range(1, 6)],
                       columns=[f"A{i}" for i in range(1, 11)]))
    print()
    print("Сумма рангов:")
    print((gen.get_sum_ranks(False)))
    print()
    print("Обобщенные ранги:")
    print((gen.get_generalized_rank(False)))
    print()
    comparisons = gen.get_paired_comparisons()
    for i in range(len(comparisons)):
        print(f"Парное сравнение (Эксперт {i + 1}):")
        print(pd.DataFrame(comparisons[i], index=[
              f"A{i + 1}" for i in range(10)], columns=[f"A{i + 1}" for i in range(10)]))
        print()
    print("Обобщенная матрица:")
    print(pd.DataFrame(gen.get_generalized_matrix_of_paired_comparisons(comparisons), index=[
          f"A{i + 1}" for i in range(10)], columns=[f"A{i + 1}" for i in range(10)]))
    print()