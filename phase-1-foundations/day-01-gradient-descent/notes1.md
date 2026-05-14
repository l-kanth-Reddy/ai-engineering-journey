PHASE 1 :DAY 1.



"How Does a Machine Actually Learn?"

&#x20;

The machine, as like humans , it also learns from experiences . 

Example : Not touching the fire, walking carefully on the slippery floor , etc .

&#x20; we experience something and from that feedback we adjust the behaviour , the machines also does the same .

Experience → Feedback → Adjust  and repeat the cycle .



"How Does a Machine Actually Learn?"

The machine actually learns by adjusting the numbers, i.e.. the start with a random numbers and calculate the  error and adjust in a way that minimises the error (correct value).



What is Gradient Descent?

&#x20;The gradient descent is reducing the wrongness of the prediction with a step by step correction.

&#x20;that is we start with a random weights(numbers)and find the error and then we adjust the weights and again repeat until we get the correct prediction.

&#x20;

High Error

&#x20;   |     \\

&#x20;   |      \\

&#x20;   |       \\        <- We start here (random weights, high error)

&#x20;   |        \\

&#x20;   |         \\

&#x20;   |          \\\_\_\_/    <- We want to reach here (low error, trained model)

&#x20;   |

Low Error

&#x20;        Weights →--------------

&#x20;

&#x20;Formula for gradient descent: 



&#x20;  new\_weight = old\_weight - (learning\_rate × gradient)



That's the entire formula for gradient descent ,

&#x20;  learning rate = the size of the change of weights( I.e.. Rate of change)..it is best to take less value like : 0.01,0.001 

&#x20;  Gradient = it indicates two things "how much to change" and "should decrease or increase the values "... we calculate mean-error by function " np.mean(error \* x) ".

&#x20;  Error = difference between the prediction and the actual output."Error= prediction - actual " . 

&#x20;  and to calculate the LOSS we use " Mean Squared Error " MSD ......."loss = np.mean(error \*\* 2)" .

&#x20;

lets see an example code to understand practically one Epoch: ...\[One epoch = one look at the data + one adjustment]





import numpy as np



x = np.array(\[1.0, 2.0, 3.0, 4.0, 5.0])

y = np.array(\[2.0, 4.0, 6.0, 8.0, 10.0])



weight = 0.0

learning\_rate = 0.01



print("=== ONE EPOCH, FULLY EXPLAINED ===\\n")



\# STEP 1: Use current weight to make predictions for ALL inputs

prediction = weight \* x

print(f"Current weight: {weight}")

print(f"Inputs x: {x}")

print(f"Predictions: {prediction}")

print(f"Actual answers y: {y}\\n")



\# STEP 2: Calculate how wrong each prediction is

error = prediction - y

print(f"Errors (prediction - actual): {error}")

print(f"The model is underestimating everywhere because weight=0\\n")



\# STEP 3: Calculate loss (single number summarizing total wrongness)

loss = np.mean(error \*\* 2)

print(f"Loss (Mean Squared Error): {loss}")

print(f"Higher loss = more wrong. We want this to decrease.\\n")



\# STEP 4: Calculate gradient

\# This answers: "if I increase weight by a tiny amount, does loss go up or down?"

gradient = np.mean(error \* x)

print(f"Gradient: {gradient}")

print(f"Negative gradient means: increasing weight will DECREASE loss")

print(f"So we should increase weight → which is what the formula will do\\n")



\# STEP 5: Update weight

old\_weight = weight

weight = weight - learning\_rate \* gradient

print(f"Old weight: {old\_weight}")

print(f"Learning rate: {learning\_rate}")

print(f"Gradient: {gradient}")

print(f"Update: {old\_weight} - ({learning\_rate} × {gradient}) = {weight:.4f}")

print(f"New weight: {weight:.4f}")

print(f"\\nWeight moved from {old\_weight} toward 2.0 ")

print(f"Now we repeat this for 99 more epochs...")



&#x20; 



OUTPUT : 

=== ONE EPOCH, FULLY EXPLAINED ===



Current weight: 0.0

Inputs x: \[1. 2. 3. 4. 5.]

Predictions: \[0. 0. 0. 0. 0.]

Actual answers y: \[ 2.  4.  6.  8. 10.]



Errors (prediction - actual): \[ -2.  -4.  -6.  -8. -10.]

The model is underestimating everywhere because weight=0



Loss (Mean Squared Error): 44.0

Higher loss = more wrong. We want this to decrease.



Gradient: -22.0

Negative gradient means: increasing weight will DECREASE loss

So we should increase weight → which is what the formula will do



Old weight: 0.0

Learning rate: 0.01

Gradient: -22.0

Update: 0.0 - (0.01 × -22.0) = 0.2200

New weight: 0.2200



Weight moved from 0.0 toward 2.0 

Now we repeat this for 99 more epochs...



\-----------------------------------------------------------------------------------------------------------



For many epochs:

\# ---  Gradient Descent From Scratch ---



\# Step 1: Import numpy

\# numpy is a library that makes math with numbers fast and easy

\# Think of it as a calculator that works on lists of numbers at once

import numpy as np



\# Step 2: Create our training data

\# x is our input, y is the correct answer

\# The rule is y = 2 \* x, but the machine doesn't know this yet

x = np.array(\[1, 2, 3, 4, 5])

y = np.array(\[2, 4, 6, 8, 10])



\# Step 3: Start with a random weight

\# The machine knows nothing. We give it a random starting guess.

\# Let's say it starts by thinking "multiply by 0" — completely wrong

weight = 0.0



\# Step 4: Set the learning rate

\# This is how big each step is

\# 0.01 is a safe, small step size for now

learning\_rate = 0.01



\# Step 5: Train for 100 rounds (called "epochs")

\# Each round = look at all data, calculate error, adjust weight

for epoch in range(100):

&#x20;   

&#x20;   # Make a prediction using current weight

&#x20;   # If weight = 0.5 and x = \[1,2,3,4,5], prediction = \[0.5, 1, 1.5, 2, 2.5]

&#x20;   prediction = weight \* x

&#x20;   

&#x20;   # Calculate the error (how wrong are we?)

&#x20;   # error = prediction - actual answer

&#x20;   # If prediction = \[0.5,1,1.5,2,2.5] and y = \[2,4,6,8,10]

&#x20;   # error = \[-1.5, -3, -4.5, -6, -7.5]  <- we're underestimating a lot

&#x20;   error = prediction - y

&#x20;   

&#x20;   # Calculate the gradient

&#x20;   # This is the math that tells us: 

&#x20;   # "which direction should we adjust weight to reduce error?"

&#x20;   # The formula np.mean(error \* x) comes from calculus (derivative of loss)

&#x20;   # Don't worry about deriving it yet — understand what it DOES

&#x20;   # It's essentially: "how much does changing weight affect our error?"

&#x20;   gradient = np.mean(error \* x)

&#x20;   

&#x20;   # Update the weight using gradient descent formula

&#x20;   # new\_weight = old\_weight - (learning\_rate \* gradient)

&#x20;   weight = weight - learning\_rate \* gradient

&#x20;   

&#x20;   # Every 10 rounds, let's see what's happening

&#x20;   if epoch % 10 == 0:

&#x20;       loss = np.mean(error \*\* 2)  # average squared error

&#x20;       print(f"Epoch {epoch}: Weight = {weight:.4f}, Loss = {loss:.4f}")



\# Final result

print(f"\\nFinal learned weight: {weight:.4f}")

print(f"Expected weight: 2.0")

print(f"If x=6, machine predicts: {weight \* 6:.2f} (correct answer: 12)")



&#x20;

&#x20;run this to understand the actuall process behind:

&#x20;for Y=2X 

Random Weight (0.0)

&#x20;      ↓

Make Prediction

&#x20;      ↓

Calculate Error (how wrong?)

&#x20;      ↓

Calculate Gradient (which direction reduces error?)

&#x20;      ↓

Update Weight (small step towards the correct value )

&#x20;      ↓

Repeat 100 times

&#x20;      ↓

Learned Weight (2.0) ← the machine figured this out itself.





\## Gradient Descent — Deep Understanding



&#x20;What I understand now:

\- Gradient = direction AND intensity of change 

\- Positive gradient → decrease weight

\- Negative gradient → increase weight  

\- Zero gradient → we've reached the correct value (stop)



\### Types of Gradient Descent:

\- Batch GD: all data at once, accurate but slow

\- Stochastic Gradient Descent: one sample, fast but noisy

\- Mini-batch GD: small batches, best of both (used in real AI)....... if the data points are more then we divide the datapoints into batches to improve system speed. 



\### Problems I now know exist:

\- Local minima: getting stuck in wrong valley( wrong values ).

\- Divergence: learning rate too high, loss explodes



&#x20;Vanishing gradient: gradients becoming too tiny in deep networks. AS we go to next layer in neural networ we have to update Gradient . so on going deep the Gradient value reduces therefore there will be no change in weights( NO step towards the correctness).



