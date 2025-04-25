import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def calculate_steps(gamma_values, num_steps, lr_rate=1e-3):
    steps_to_disappearance = []
    for gamma in gamma_values:
        regress_reward = 0
        steps = 0
        rewards = np.round(np.linspace(5e-5, 2e-1, num_steps), 10)  # Round rewards to 5 decimal places
        for reward in rewards:
            regress_reward = reward
            steps = 0
            while abs(regress_reward) > 1e-07:
                regress_reward = gamma * regress_reward * lr_rate
                steps += 1
            steps_to_disappearance.append((reward, gamma, steps))
    return steps_to_disappearance


gamma_values = [0.5, 0.70, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95]  # Test gamma values from 0 to 1
num_steps = 10

steps_to_disappearance = calculate_steps(gamma_values, num_steps, lr_rate=1.)

# Create a DataFrame with the reward values and the steps to disappearance
df = pd.DataFrame(steps_to_disappearance, columns=['Reward', 'Gamma', 'Steps'])

# Plot the results
sns.set_context("paper")
sns.set_style("whitegrid")
plt.figure(figsize=(24, 16))
ax = sns.barplot(x='Reward', y='Steps', hue='Gamma', data=df)
for i, p in enumerate(ax.patches):
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), str(int(p.get_height())),
            ha="center", va="bottom", color="black", size=8)
    gamma_index = i // num_steps
    if gamma_index < len(gamma_values):
        gamma_value = gamma_values[gamma_index]
        ax.text(p.get_x() + p.get_width() / 2., -0.1, f"{gamma_value}",
                ha="center", va="bottom", color="black", size=8)
plt.xlabel('Reward Value')
plt.ylabel('Steps to Disappearance')
plt.title('Steps to Disappearance vs Reward Value')
plt.grid(True)
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.1))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
