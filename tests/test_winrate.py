import matplotlib.pyplot as plt
import numpy as np


# Функция для расчета score
def calculate_score(win_rate, weight=0.55):
    if win_rate > 0.:
        if win_rate > 0.51:
            score = np.exp(win_rate) * weight * 1.2
        else:
            score = np.exp(win_rate) * (weight ** 2)
    else:
        score = -np.exp(-win_rate) * weight
    return score * 0.001


# Создаем массив значений win_rate от 0.1 до 1.0 с шагом 0.01
win_rates = np.arange(-0.5, 0.75, 0.01)
scores = [calculate_score(w) for w in win_rates]

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(win_rates, scores, label='Score')
plt.xlabel('Win Rate')
plt.ylabel('Score')
plt.title('Зависимость Score от Win Rate')
plt.legend()
plt.grid(True)
plt.show()
