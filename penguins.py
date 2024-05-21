# Linear Regression Model using TensorFlow
# @author Ryan Magnuson rmagnuson@westmont.edu

# Setup
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import pandas as pd

# Seed Control + Data Gen.
# np.random.seed(256) # TODO: Uncomment this line to make results consistent!

learning_rate = 0.01
training_epochs = 100 # reps

X_train = np.linspace(0, 10, 100)
y_train = X_train + np.random.normal(0,1,100)

# Display initial graph
plt.scatter(X_train, y_train)
plt.savefig("dataset.png")
plt.close()

# Weights
weight = tf.Variable(0.)
bias = tf.Variable(0.)

# LINEAR REGRESSION #
def linreg(x):
  ##
  # Linear Regression equation for ML
  # @param : x
  # @return : y
  return weight*x + bias # = y

def squared_error(y_pred, y_true):
  ##
  # MSE Loss function
  # @param : y_pred
  # @param : y_true
  # @return : MSE
  return tf.reduce_mean(tf.square(y_pred - y_true))

# Train the LRM
for epoch in range(training_epochs):

  # Compute loss with Gradient Tape
  with tf.GradientTape() as tape:
    y_predicted = linreg(X_train)
    loss = squared_error(y_predicted, y_train)

    # Gradients
    gradients = tape.gradient(loss, [weight, bias])

    # Adjust Weights
    weight.assign_sub(gradients[0]*learning_rate)
    bias.assign_sub(gradients[1]*learning_rate)

    # Output
    print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")

# Final Output
print("\nWeight: %s | Bias: %s" % (weight.numpy(), bias.numpy()))

# Plot the LRM Line
plt.scatter(X_train, y_train)
plt.plot(X_train, linreg(X_train), 'r')
plt.savefig("linear_regression_result.png")
plt.close()
