# SRIP-IITGN-2022---Machine-Learning---Prof.-Nipun-Batra
### This is Anushka Rajput
### Reg Id: 21149376
### IP0NB0000010 - Programatically Understanding ML Faculty Advisor: Nipun Batra
### This submission consist of solution for task 3 and task 4 

Task 3 : Implement two hidden layers neural network classifier from scratch in JAX

* Two hidden layers here means (input - hidden1 - hidden2 - output).
* You must not use flax, optax, or any other library for this task.
* Use MNIST dataset with 80:20 train:test split.
* Manually optimize the number of neurons in hidden layers.
* Use gradient descent from scratch to optimize your network. You should use the Pytree concept of JAX to do this elegantly.
* Plot loss v/s iterations curve with matplotlib.
* Evaluate the model on test data with various classification metrics and briefly discuss their implications.


Task 4 : Bayesian Linear Regression from scratch with BlackJAX 

* Implement Bayesian Linear Regression from scratch with any appropriate sampling method in BlackJAX.
* Create your own 1d linear dataset with added noise.
* Plot the learned predictive mean and 2 standard deviations around the mean like the below plot.
