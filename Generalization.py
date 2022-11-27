import numpy as np
import pandas as pd


class Generalization:
    def __init__(self, matrix: np.array, **kwargs) -> None:
        """
        Args:
            matrix (np.array): Матрица мнения экспертов.
        """
        self.matrix = matrix
        self.competence = kwargs.get("competence")

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

    def get_sum_ranks(self, flag: bool = False) -> np.array:
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

    def get_paired_comparisons(self) -> np.array:
        """Метод для получения матриц парных сравнений.

        Returns:
            np.array: Матрица парных сравнений.
        """
        paired_comparisons = []
        for i in range(len(self.matrix)):
            comparison = np.zeros((len(self.matrix[0]), len(self.matrix[0])), dtype=int)
            for j in range(len(self.matrix[0])):
                for k in range(len(self.matrix[0])):
                    if self.matrix[i][j] > self.matrix[i][k]:
                        comparison[j][k] = 0
                    elif self.matrix[i][j] <= self.matrix[i][k]:
                        comparison[j][k] = 1
            paired_comparisons.append(comparison)
        return np.array(paired_comparisons)

    def get_generalized_matrix_of_paired_comparisons(self, paired_comparisons: np.array):
        comparison = np.zeros((len(self.matrix[0]), len(self.matrix[0])), dtype=int)
        for j in range(len(self.matrix[0])):
            for k in range(len(self.matrix[0])):
                sum = 0
                for i in range(len(self.matrix)):
                    sum += paired_comparisons[i][j][k]
                if sum > 2:
                    comparison[j][k] = 1
                else:
                    comparison[j][k] = 0
        return comparison