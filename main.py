import numpy as np
import pandas as pd
from scipy.stats import rankdata


class Generalization:
    def __init__(self, matrix: np.array, competence: np.array) -> None:
        """
        Args:
            matrix (np.array): Матрица мнения экспертов.
            competence (np.array): Матрица компетентности экспертов.
        """
        self.matrix = matrix
        self.competence = competence

    def get_generalized_rank(self, flag: bool = False) -> np.array:
        """Метод для получения рангов по методу обобщенного ранжирования.
        Args:
            flag (bool): Флаг для использования компетентности экспертов. По умолчанию False.
        Returns:
            np.array: Матрица обобщенных рангов.
        """
        amount_rank = self.get_sum_ranks(flag)
        generalized_rank = [0]*len(amount_rank)
        d = {}
        for i, value in enumerate(amount_rank, start=1):
            d[i] = value
        d = sorted(d.items(), key=lambda x: x[1])
        for i, value in enumerate(d, start=1):
            rank = 0
            count = 0
            for j in range(len(d)):
                if d[j][1] == value[1]:
                    rank += j+1
                    count += 1
            rank /= count
            generalized_rank[value[0]-1] = rank
        return generalized_rank

    def get_sum_ranks(self, flag: bool=False):
        """Метод для получения суммы рангов.

        Args:
            flag (bool): Флаг для использования компетентности экспертов. По умолчанию False.

        Returns:
            np.array: Матрица сумм рангов.
        """
        sum_ranks = []
        for i in range(len(self.matrix[0])):
            sum = 0
            for j in range(len(self.matrix)):
                if flag:
                    sum += self.matrix[j][i] * self.competence[j]
                else:
                    sum += self.matrix[j][i]
            sum_ranks.append(sum)
        return np.array(sum_ranks)


if __name__ == "__main__":
    matrix = np.array([
        [1, 1, 6, 8, 10, 2, 6, 5, 1, 4],
        [10, 8, 5, 5, 5, 4, 6, 6, 9, 8],
        [2, 3, 9, 1, 5, 2, 10, 8, 6, 5],
        [10, 9, 9, 3, 6, 1, 5, 4, 2, 2],
        [5, 7, 8, 4, 7, 5, 9, 9, 3, 7]
    ]
    )
    competence = np.array([.086921, .011919, .43016, .10141, .36959])
    gen = Generalization(matrix, competence)
    print("2. Обобщить мнения экспертов, полученные непосредственной оценкой по балльной шкале:")
    print("а) без учёта компетентности экспертов, б) с учётом")
    print("Матрица оценок экспертов:")
    print(pd.DataFrame(matrix, index=[f"Эксперт {i}" for i in range(1, 6)],
                       columns=[f"A{i}" for i in range(1, 11)]))
    print("Компетентность экспертов:")
    print(pd.DataFrame(competence))
    print("Сумма рангов:")
    print(gen.get_sum_ranks(False))
    print("Обобщенные ранги:")
    print(gen.get_generalized_rank(False))
    print("Сумма рангов с учётом компетентности:")
    print(gen.get_sum_ranks(True))
    print("Обобщенные ранги с учётом компетентности:")
    print(gen.get_generalized_rank(True))
