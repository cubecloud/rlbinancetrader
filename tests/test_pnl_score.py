import matplotlib.pyplot as plt
import numpy as np


# Функция для расчета score
def pnl_score(pnl, buy_and_hold_pnl: float = 0.04, weight: float = 0.8):
    if pnl > .0:
        if pnl > buy_and_hold_pnl:
            score = np.exp(pnl)
        else:
            score = np.exp(pnl) * weight
    else:
        score = -(np.exp(-pnl) * (weight ** 3))
    return score * 0.1


# Создаем массив значений win_rate от 0.1 до 1.0 с шагом 0.01
pnl = np.arange(-0.2, 0.21, 0.01)
scores = [pnl_score(w) for w in pnl]
print(np.exp(0))
# Строим график
plt.figure(figsize=(10, 6))
plt.plot(pnl, scores, label='Score')
plt.xlabel('PnL')
plt.ylabel('Score')
plt.title('Зависимость Score от PnL')
plt.legend()
plt.grid(True)
plt.show()
