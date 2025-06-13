import timeit
from collections import Counter
import numpy as np


def count_numbers_with_loops(s):
    numbers = list(map(int, s.split(',')))

    count_positive = 0
    count_negative = 0
    count_zero = 0

    for num in numbers:
        if num > 0:
            count_positive += 1
        elif num < 0:
            count_negative += 1
        else:
            count_zero += 1

    return count_positive, count_negative, count_zero


def count_numbers_with_filter(s):
    numbers = list(map(int, s.split(',')))

    count_positive = len(list(filter(lambda x: x > 0, numbers)))
    count_negative = len(list(filter(lambda x: x < 0, numbers)))
    count_zero = len(list(filter(lambda x: x == 0, numbers)))

    return count_positive, count_negative, count_zero


def count_numbers_with_comprehensions(s):
    numbers = list(map(int, s.split(',')))

    count_positive = sum(1 for num in numbers if num > 0)
    count_negative = sum(1 for num in numbers if num < 0)
    count_zero = sum(1 for num in numbers if num == 0)

    return count_positive, count_negative, count_zero


def count_numbers_with_counter(s):
    numbers = list(map(int, s.split(',')))

    # Классифицируем числа
    categories = []
    for num in numbers:
        if num > 0:
            categories.append('positive')
        elif num < 0:
            categories.append('negative')
        else:
            categories.append('zero')

    # Используем Counter для подсчета количества каждой категории
    counter = Counter(categories)

    count_positive = counter['positive']
    count_negative = counter['negative']
    count_zero = counter['zero']

    return count_positive, count_negative, count_zero


def count_numbers_with_numpy(s):
    numbers = np.asarray(list(map(int, s.split(','))))

    count_positive = np.sum(numbers > 0)
    count_negative = np.sum(numbers < 0)
    count_zero = np.sum(numbers == 0)

    return count_positive, count_negative, count_zero


def count_numbers_with_numpy_optimized(s):
    numbers = np.asarray(list(map(int, s.split(','))))

    count_positive = np.count_nonzero(numbers > 0)
    count_negative = np.count_nonzero(numbers < 0)
    count_zero = numbers.shape[0] - count_positive - count_negative

    return count_positive, count_negative, count_zero


# Пример строки с числами
numbers_string = "1, -2, 0, 3, 4, -5, 0" * 1000  # Увеличиваем размер для более точных замеров
number = 10000

# Измеряем время выполнения первой версии функции
time_loops = timeit.timeit(lambda: count_numbers_with_loops(numbers_string), number=number)
print(f"Время выполнения с использованием циклов: {time_loops:.6f} секунд")

# Измеряем время выполнения второй версии функции
time_filter = timeit.timeit(lambda: count_numbers_with_filter(numbers_string), number=number)
print(f"Время выполнения с использованием filter: {time_filter:.6f} секунд")

# Измеряем время выполнения третьей версии функции
time_comprehensions = timeit.timeit(lambda: count_numbers_with_comprehensions(numbers_string), number=number)
print(f"Время выполнения с использованием списковых включений: {time_comprehensions:.6f} секунд")

# Измеряем время выполнения четвертой версии функции
time_counter = timeit.timeit(lambda: count_numbers_with_counter(numbers_string), number=number)
print(f"Время выполнения с использованием Counter: {time_counter:.6f} секунд")

# Измеряем время выполнения пятой версии функции
time_numpy = timeit.timeit(lambda: count_numbers_with_numpy(numbers_string), number=number)
print(f"Время выполнения с использованием numpy: {time_numpy:.6f} секунд")

# Измеряем время выполнения шестой версии функции
time_numpy_optimized = timeit.timeit(lambda: count_numbers_with_numpy_optimized(numbers_string), number=number)
print(f"Время выполнения с использованием оптимизированного numpy: {time_numpy_optimized:.6f} секунд")
