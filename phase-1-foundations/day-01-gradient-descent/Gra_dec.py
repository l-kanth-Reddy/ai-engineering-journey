# --- Day 1: Gradient Descent From Scratch ---

# Step 1: Import numpy
# numpy is a library that makes math with numbers fast and easy
# Think of it as a calculator that works on lists of numbers at once
import numpy as np

# Step 2: Create our training data
# x is our input, y is the correct answer
# The rule is y = 2 * x, but the machine doesn't know this yet
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Step 3: Start with a random weight
# The machine knows nothing. We give it a random starting guess.
# Let's say it starts by thinking "multiply by 0" — completely wrong
weight = 0.0

# Step 4: Set the learning rate
# This is how big each step is
# 0.01 is a safe, small step size for now
learning_rate = 0.01

# Step 5: Train for 100 rounds (called "epochs")
# Each round = look at all data, calculate error, adjust weight
for epoch in range(100):
    
    # Make a prediction using current weight
    # If weight = 0.5 and x = [1,2,3,4,5], prediction = [0.5, 1, 1.5, 2, 2.5]
    prediction = weight * x
    
    # Calculate the error (how wrong are we?)
    # error = prediction - actual answer
    # If prediction = [0.5,1,1.5,2,2.5] and y = [2,4,6,8,10]
    # error = [-1.5, -3, -4.5, -6, -7.5]  <- we're underestimating a lot
    error = prediction - y
    
    # Calculate the gradient
    # This is the math that tells us: 
    # "which direction should we adjust weight to reduce error?"
    # The formula np.mean(error * x) comes from calculus (derivative of loss)
    # Don't worry about deriving it yet — understand what it DOES
    # It's essentially: "how much does changing weight affect our error?"
    gradient = np.mean(error * x)
    
    # Update the weight using gradient descent formula
    # new_weight = old_weight - (learning_rate * gradient)
    weight = weight - learning_rate * gradient
    
    # Every 10 rounds, let's see what's happening
    if epoch % 10 == 0:
        loss = np.mean(error ** 2)  # average squared error
        print(f"Epoch {epoch}: Weight = {weight:.4f}, Loss = {loss:.4f}")

# Final result
print(f"\nFinal learned weight: {weight:.4f}")
print(f"Expected weight: 2.0")
print(f"If x=6, machine predicts: {weight * 6:.2f} (correct answer: 12)")
