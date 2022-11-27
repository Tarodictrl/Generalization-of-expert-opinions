from Generalization import Generalization
import numpy as np
import pandas as pd

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
    gen = Generalization(matrix, competence=competence)
    print("2. Обобщить мнения экспертов, полученные непосредственной оценкой по балльной шкале:")
    print("а) без учёта компетентности экспертов, б) с учётом")
    print("Матрица оценок экспертов:")
    print(pd.DataFrame(matrix, index=[f"Эксперт {i}" for i in range(1, 6)],
                       columns=[f"A{i}" for i in range(1, 11)]))
    print()
    print("Компетентность экспертов:")
    print(pd.DataFrame(competence, index=[f"Эксперт {i}" for i in range(1, 6)], columns=["Компетентность"]))
    print()
    print("Сумма рангов без учёта компетентности экспертов:")
    print(gen.get_sum_ranks(False))
    print()
    print("Обобщенные ранги без учёта компетентности экспертов:")
    print(gen.get_generalized_rank(False))
    print()
    print("Сумма рангов с учётом компетентности:")
    print(gen.get_sum_ranks(True))
    print()
    print("Обобщенные ранги с учётом компетентности:")
    print(gen.get_generalized_rank(True))
    