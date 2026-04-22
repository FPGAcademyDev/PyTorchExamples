# Run on
# https://app.coderpad.io/sandbox or
# https://colab.research.google.com/
# Visualize result at https://www.desmos.com/calculator
# Compare print out to y = sin(x)
# Further point of interest to compare is y = x - x^3/3! +  x^5/5! -  x^7/7! +  x^9/9! 

import math
import torch

N, epochs = 2000, 2000

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, N)
y = torch.sin(x)

# Randomly initialize weights
w = torch.randn(4, 1, requires_grad=True)

learning_rate = 1e-6
for t in range(epochs):
    # Forward pass: compute predicted y
    # y = w_0 + w_1 x + w_2 x^2 + w_3 x^3
    y_hat = w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3

    # Compute and print loss
    loss = torch.square(y_hat - y).mean()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of the weights with respect to loss
    grad_y_hat = 2.0 * (y_hat - y)
    grad_w0 = grad_y_hat.sum()
    grad_w1 = (grad_y_hat * x).sum()
    grad_w2 = (grad_y_hat * x ** 2).sum()
    grad_w3 = (grad_y_hat * x ** 3).sum()

    # Update weights
    with torch.no_grad():
        w[0] -= learning_rate * grad_w0
        w[1] -= learning_rate * grad_w1
        w[2] -= learning_rate * grad_w2
        w[3] -= learning_rate * grad_w3

print(f'Result: y = {w[0].item()} + {w[1].item()} x + {w[2].item()} x^2 + {w[3].item()} x^3')
