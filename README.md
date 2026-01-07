1. Perceptron Implementation (perceptron.ipynb)

This notebook demonstrates how a single-layer perceptron finds a linear decision boundary to separate two classes of data.

Logic: I implemented the Perceptron Learning Rule, an iterative algorithm that adjusts weights based on classification errors.

The Process: * The model starts with weights and bias set to zero.

It scans every data point; if a point is misclassified, the weights are nudged toward that point to correct the error.

This continues until a Decision Boundary (a straight line) perfectly separates the classes.

Experimentation: I tested the model on linearly separable data (where it converges perfectly) and noisy data (where the overlap prevents a perfect split), illustrating the limitations of linear classifiers.

2.Linear Regression Implementation

This project implements basic Linear Regression from scratch using PyTorch. It explores two ways to find the best-fitting line for a 10-feature dataset.

Core Features

Closed-Form Solution: Calculates the exact weights using the mathematical Normal Equation $(w = (X^T X)^{-1} X^T y)$.

Gradient Descent: An iterative algorithm that slowly "walks" toward the minimum error by updating weights using the loss gradient.

L2 Loss: Uses Mean Squared Error (MSE) to measure how far predictions are from the actual values.

How it Works

Data Split: The dataset is divided 80/20 for training and testing.

Training: The model updates weights using the rule: $w = w - \eta \cdot \nabla L$.

Convergence: The training stops automatically when the loss stops changing (less than $1e-15$).

Visualization: A plot of the Test Error shows how the model improves over time.

Key Functions

w_closed_form: The direct math solution.

l2_loss: Measures the model's error.

train_model: The loop that performs Gradient Descent.
