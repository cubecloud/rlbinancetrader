import matplotlib.pyplot as plt
import numpy as np


# Define the sigmoid function
def sigmoid(x, norm_factor=96):
    """
    Computes the sigmoid function with a given normalization value.

    Parameters:
    x (array-like): Input values to compute the sigmoid function on.
    norm_factor (int, optional): The scaling factor for the input values. Defaults to 96.

    Returns:
    array-like: The computed sigmoid values.
    """
    return 1 / (1 + np.exp(-(x / norm_factor)))


# Set the normalization value globally
normalization_value = 96

# Generate test data from 0 to 450
x_values = np.arange(0, 450)

# Compute the sigmoid values using the specified normalization value
y_values = sigmoid(x_values, norm_factor=normalization_value)

# Plot the sigmoid curve
plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values, label='Sigmoid')
plt.xlabel('X Values')
plt.ylabel('Sigmoid Output')
plt.title(f'Sigmoid Function with Normalization Value {normalization_value}')  # Use declared variable
plt.grid(True)
plt.legend()
plt.show()
