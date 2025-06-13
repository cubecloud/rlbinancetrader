import timeit
import numpy as np
from numba import njit

# Вариант 1: Использование циклов
def count_numbers_list(numbers):
    positive_count = 0
    negative_count = 0
    zero_count = 0
    for number in numbers:
        if number > 0:
            positive_count += 1
        elif number < 0:
            negative_count += 1
        else:
            zero_count += 1
    return positive_count, negative_count, zero_count

# Вариант 2: Использование словаря
def count_numbers_dict(numbers):
    count_dict = {'positive': 0, 'negative': 0, 'zero': 0}
    for number in numbers:
        if number > 0:
            count_dict['positive'] += 1
        elif number < 0:
            count_dict['negative'] += 1
        else:
            count_dict['zero'] += 1
    return count_dict

# Вариант 3: Использование NumPy
def count_numbers_numpy(numbers):
    positive_count = np.sum(numbers > 0)
    negative_count = np.sum(numbers < 0)
    zero_count = numbers.shape[0] - positive_count - negative_count
    return positive_count, negative_count, zero_count

# Вариант 4: Использование Numba
@njit
def count_numbers_numba(numbers):
    positive_count = 0
    negative_count = 0
    zero_count = 0
    for number in numbers:
        if number > 0:
            positive_count += 1
        elif number < 0:
            negative_count += 1
        else:
            zero_count += 1
    return positive_count, negative_count, zero_count

# Массив чисел для тестирования
numbers = np.random.randint(-100, 100, size=5000)

# Измерение времени выполнения каждой функции
time_loops = timeit.timeit(lambda: count_numbers_list(numbers), number=10000)
time_dict = timeit.timeit(lambda: count_numbers_dict(numbers), number=10000)
time_numpy = timeit.timeit(lambda: count_numbers_numpy(numbers), number=10000)
time_numba = timeit.timeit(lambda: count_numbers_numba(numbers), number=10000)

# Вывод результатов
print(f"Время выполнения с использованием циклов: {time_loops:.6f} секунд")
print(f"Время выполнения с использованием словаря: {time_dict:.6f} секунд")
print(f"Время выполнения с использованием NumPy: {time_numpy:.6f} секунд")
print(f"Время выполнения с использованием Numba: {time_numba:.6f} секунд")