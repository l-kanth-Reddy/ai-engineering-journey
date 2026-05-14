import numpy as np
import matplotlib.pyplot as plt

# Create a range of inputs from -6 to +6
x = np.linspace(-6, 6, 300)

# --- Define each activation function ---

def step_function(x):
    # If x >= 0, return 1. Else return 0.
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    # Squashes everything between 0 and 1
    return 1 / (1 + np.exp(-x))

def relu(x):
    # Maximum of 0 and x. Negatives become 0.
    return np.maximum(0, x)

def tanh(x):
    # Like sigmoid but squashes between -1 and +1
    # Better than sigmoid for hidden layers (centered at 0)
    return np.tanh(x)

# --- Plot them all ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Activation Functions', fontsize=16, fontweight='bold')

# Step Function
axes[0,0].plot(x, step_function(x), 'b-', linewidth=2)
axes[0,0].set_title('Step Function\n(historical, not used today)')
axes[0,0].axhline(y=0, color='k', linewidth=0.5)
axes[0,0].axvline(x=0, color='k', linewidth=0.5)
axes[0,0].set_ylim(-0.2, 1.2)
axes[0,0].grid(True, alpha=0.3)

# Sigmoid
axes[0,1].plot(x, sigmoid(x), 'g-', linewidth=2)
axes[0,1].set_title('Sigmoid\n(output layer, binary classification)')
axes[0,1].axhline(y=0, color='k', linewidth=0.5)
axes[0,1].axvline(x=0, color='k', linewidth=0.5)
axes[0,1].grid(True, alpha=0.3)

# ReLU
axes[1,0].plot(x, relu(x), 'r-', linewidth=2)
axes[1,0].set_title('ReLU\n(hidden layers, most common today)')
axes[1,0].axhline(y=0, color='k', linewidth=0.5)
axes[1,0].axvline(x=0, color='k', linewidth=0.5)
axes[1,0].grid(True, alpha=0.3)

# Tanh
axes[1,1].plot(x, tanh(x), 'm-', linewidth=2)
axes[1,1].set_title('Tanh\n(hidden layers, RNNs)')
axes[1,1].axhline(y=0, color='k', linewidth=0.5)
axes[1,1].axvline(x=0, color='k', linewidth=0.5)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Graph saved as activation_functions.png")