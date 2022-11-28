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
            flag (bool): Флаг для использования компетентности экспертов. 
            По умолчанию False.
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
            flag (bool): Флаг для использования компетентности экспертов. 
            По умолчанию False.

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
            comparison = np.zeros(
                (len(self.matrix[0]), len(self.matrix[0])), dtype=int)
            for j in range(len(self.matrix[0])):
                for k in range(len(self.matrix[0])):
                    if self.matrix[i][j] < self.matrix[i][k]:
                        comparison[j][k] = 0
                    elif self.matrix[i][j] >= self.matrix[i][k]:
                        comparison[j][k] = 1
            paired_comparisons.append(comparison)
        return np.array(paired_comparisons)

    def get_generalized_matrix_of_paired_comparisons(self, paired_comparisons: np.array) -> np.zeros:
        """Метод для получения обобщенной матрицы парных сравнений.

        Args:
            paired_comparisons (np.array): Матрицы парных сравнений.

        Returns:
            np.zeros: Матрица обобщенных парных сравнений.
        """
        comparison = np.zeros(
            (len(self.matrix[0]), len(self.matrix[0])), dtype=int)
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

    def get_min_max_arrays(self) -> tuple:
        """Метод для получения массивов максимальных и минимальных рангов.

        Returns:
            tuple: Массивы максимальных и минимальных рангов.
        """
        dt = pd.DataFrame(self.matrix)
        min_array = []
        max_array = []
        for i in range(len(dt.columns)):
            min_array.append(dt[i].min())
            max_array.append(dt[i].max())
        return min_array, max_array

    def get_normalized_matrix(self) -> np.array:
        """Метод для получения нормализованной матрицы.

        Returns:
            np.array: Нормализованная матрица.
        """
        matrix = np.zeros((len(self.matrix), len(self.matrix[0])))
        min_array, max_array = self.get_min_max_arrays()
        for i in range(len(self.matrix[0])):
            for j in range(len(self.matrix)):
                if i % 2 == 0:
                    matrix[j][i] = (max_array[i] - self.matrix[j]
                                    [i]) / (max_array[i] - min_array[i])
                else:
                    matrix[j][i] = self.matrix[j][i] / max_array[i]
        return np.array(matrix)

    def get_additive_convolution(self, normalized_matrix: np.array,
                                 method: str = "equivalent", **kwargs) -> np.array:
        """Метод для получения матрицы суммарной свертки.

        Args:
            normalized_matrix (np.array): Нормализованная матрица.
            method (str, optional): Метод для получения матрицы суммарной свертки. По умолчанию "equivalent".
        Raises:
            ValueError: Если метод не найден.

        Returns:
            np.array: Матрица суммарной свертки.
        """
        if method not in ["equivalent", "specified"]:
            raise ValueError("Неверный метод!")
        convolution = []
        if method == "equivalent":
            for i in range(len(self.matrix)):
                print(normalized_matrix[i])
                convolution.append(
                    sum(normalized_matrix[i])/len(normalized_matrix[i]))
        elif method == "specified":
            weights = kwargs.get("weights")
            for i in range(len(self.matrix)):
                sums = 0
                for j in range(len(self.matrix[0])):
                    sums += normalized_matrix[i][j] * weights[j]
                convolution.append(sums)
        return np.array(convolution)

    def get_multiplicative_convolution(self, normalized_matrix: np.array,
                                       method: str = "equivalent", **kwargs) -> np.array:
        """Метод для получения матрицы мультипликативной свертки.

        Args:
            normalized_matrix (np.array): Нормализованная матрица.
            method (str, optional): Метод для получения матрицы мультипликативной свертки. По умолчанию "equivalent".

        Raises:
            ValueError: Если метод не найден.

        Returns:
            np.array: Матрица мультипликативной свертки.
        """
        if method not in ["equivalent", "specified"]:
            raise ValueError("Неверный метод!")
        convolution = []
        if method == "equivalent":
            for i in range(len(self.matrix)):
                convolution.append(
                    np.prod(normalized_matrix[i])**(1/len(normalized_matrix[i])))
        elif method == "specified":
            for i in range(len(self.matrix)):
                sums = 1
                for j in range(len(self.matrix[0])):
                    sums *= normalized_matrix[i][j] ** kwargs.get("weights")[j]
                convolution.append(sums)
        return np.array(convolution)

    def get_perfect_point(self, normalized_matrix: np.array, 
                          method: str = "equivalent", **kwargs) -> np.array:
        if method not in ["equivalent", "specified"]:
            raise ValueError("Неверный метод!")
        perfect_point = []
        if method == "equivalent":
            for i in range(len(self.matrix)):
                sum = 0
                for j in range(len(self.matrix[0])):
                    sum += (1 - normalized_matrix[i][j])**2
                perfect_point.append(sum/len(self.matrix[0]))
        elif method == "specified":
            weights = kwargs.get("weights")
            for i in range(len(self.matrix)):
                sum = 0
                for j in range(len(self.matrix[0])):
                    sum += weights[j] * (1 - normalized_matrix[i][j])**2
                perfect_point.append(sum)
        return np.array(perfect_point)