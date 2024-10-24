import numpy as np
import matplotlib.pyplot as plt

# Define the combined reward function
def combined_reward(previous_pnl, previous_buy_and_hold_pnl):
    conditions = [(previous_pnl > 0) & (previous_buy_and_hold_pnl > 0),
                  (previous_pnl > 0) & (previous_buy_and_hold_pnl < 0),
                  (previous_pnl < 0) & (previous_buy_and_hold_pnl > 0),
                  (previous_pnl < 0) & (previous_buy_and_hold_pnl < 0)]
    choices = [(previous_pnl + previous_buy_and_hold_pnl) * 0.1,
               (previous_pnl - previous_buy_and_hold_pnl) * 0.1,
               (previous_buy_and_hold_pnl - previous_pnl) * 0.1,
               (previous_pnl + previous_buy_and_hold_pnl) * 0.5]  # <--- modified this line
    reward = np.select(conditions, choices, default=0)
    return reward

# Create a grid of possible values for previous_pnl and previous_buy_and_hold_pnl
previous_pnl_values = np.linspace(-0.03, 0.03, 1000)
previous_buy_and_hold_pnl_values = np.linspace(-0.03, 0.03, 1000)
previous_pnl_grid, previous_buy_and_hold_pnl_grid = np.meshgrid(previous_pnl_values, previous_buy_and_hold_pnl_values)

# Calculate the combined reward for each point in the grid
combined_reward_grid = combined_reward(previous_pnl_grid, previous_buy_and_hold_pnl_grid)

# Find the maximum combined reward and the corresponding rules
max_reward = np.max(combined_reward_grid)
max_reward_index = np.unravel_index(np.argmax(combined_reward_grid), combined_reward_grid.shape)
best_previous_pnl = previous_pnl_values[max_reward_index[1]]
best_previous_buy_and_hold_pnl = previous_buy_and_hold_pnl_values[max_reward_index[0]]

# Print the best rules
print(f"Best rules: previous_pnl = {best_previous_pnl:.4f}, previous_buy_and_hold_pnl = {best_previous_buy_and_hold_pnl:.4f}")
print(f"Maximum combined reward: {max_reward:.4f}")

# Plot the combined reward function
fig, ax = plt.subplots()
contour = ax.contourf(previous_pnl_grid, previous_buy_and_hold_pnl_grid, combined_reward_grid, levels=40)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("Combined Reward")

# Add a function to display the data values under the cursor
def update(event):
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            i = np.argmin(np.abs(previous_pnl_values - x))
            j = np.argmin(np.abs(previous_buy_and_hold_pnl_values - y))
            reward = combined_reward_grid[j, i]
            ax.set_title(f"Combined Reward: {reward:.4f} at ({x:.4f}, {y:.4f})")
    fig.canvas.draw_idle()

# Connect the update function to the motion event
fig.canvas.mpl_connect("motion_notify_event", update)

# Show the plot
plt.xlabel("Previous PNL")
plt.ylabel("Previous Buy and Hold PNL")
plt.title("Combined Reward Function")
plt.show()
