# Find the steady state of the following Markov chain:

import numpy as np

# Lecture Notes
transition_matrix = np.array(
    [
        [0.4, 0.6, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.3, 0, 0.7, 0],
        [0, 0, 0.1, 0.3, 0.6],
        [0, 0.3, 0, 0.5, 0.2],
    ]
)
# This is the Singular Matrix
# We add a condition that Sum (pi_i) = 1 and solve for the steady state

condition = np.ones(5)

steady_state = np.linalg.solve(transition_matrix, condition)
print(steady_state)

# P^T * pi = pi and resolve for pi

steady_state = np.linalg.solve(transition_matrix.T, condition)
print(steady_state)

# Expected Times to reach the critical state
# t_i = E[T | X_0 = i] or in English, the expected time to reach the critical state starting from i

# i != n, t_i = 1 + ( P_ij * t_j for all j != n)

# i = n, t_n = 0

# We can solve the system of equations to get each t_i

# t = (I - P_nn)^-1 * (1 + P_nj * t_j for all j != n)

# We can solve this system of linear equations using matrix inversion

# Example code:

# Define the transition matrix
transition_matrix = np.array(
    [
        [0.4, 0.6, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.3, 0, 0.7, 0],
        [0, 0, 0.1, 0.3, 0.6],
        [0, 0.3, 0, 0.5, 0.2],
    ]
)

# Define the identity matrix
I = np.eye(5)

# Define the matrix (I - P_nn)
P_nn = np.diag(transition_matrix)

# Define the matrix (I - P_nn)^-1
P_nn_inv = np.linalg.inv(I - P_nn)

# Define the matrix (1 + P_nj * t_j for all j != n)
P_nj = transition_matrix - P_nn

# t is not defined in the question, so we need to define it
# Define the matrix t
t = np.zeros(5)

# Solve for t
t = P_nn_inv @ (1 + P_nj @ t)

# Solve for t
t = np.linalg.solve(P_nn_inv, 1 + P_nj @ t)

print(t)

"""
Write a python code to find the probability of being in various states after a long amount of time.
"""
transition_matrix = np.array(
    [
        [0.4, 0.6, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.3, 0, 0.7, 0],
        [0, 0, 0.1, 0.3, 0.6],
        [0, 0.3, 0, 0.5, 0.2],
    ]
)

probability_of_being_in_state = np.linalg.matrix_power(transition_matrix, 1000)

# Verify the steady state by analytically solving the system of equations

steady_state = np.linalg.solve(transition_matrix.T, np.ones(5))

print(f"The steady state is {steady_state}")

print(f"The probability of being in state is {probability_of_being_in_state}")

""" Q2
Same markov chain

Start from state 1.

What is the expected time to reach state 5?
"""

# We can use the same transition matrix as before

# Define the transition matrix
transition_matrix = np.array(
    [
        [0.4, 0.6, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.3, 0, 0.7, 0],
        [0, 0, 0.1, 0.3, 0.6],
        [0, 0.3, 0, 0.5, 0.2],
    ]
)

# Define the identity matrix
I = np.eye(
    5
)  # Explore the identity matrix, it is a square matrix with ones on the diagonal and zeros elsewhere

# Define the matrix (I - P_nn)
P_nn = np.diag(
    transition_matrix
)  # diag is to get the diagonal elements of the matrix, that is the probability of staying in the same state

# Define the matrix (I - P_nn)^-1
P_nn_inv = np.linalg.inv(
    I - P_nn
)  # Explore the inverse of a matrix, it is a matrix that when multiplied by the original matrix, results in the identity matrix

# Define the matrix (1 + P_nj * t_j for all j != n)
P_nj = (
    transition_matrix - P_nn
)  # Why do we need to minus P_nn from the transition matrix?
# Explore the matrix P_nj, it is the transition matrix minus the diagonal elements

# Define the matrix t
t = np.zeros(5)

# Solve for t
t = P_nn_inv @ (1 + P_nj @ t)

# The expected time to reach state 5 is t[4]
print(f"The expected time to reach state 5 is {t[4]:.3f}")

# Why is it also called the heat map?
# The heat map is a matrix that shows the probability of being in a state after a long amount of time
# The heat map is the steady state of the transition matrix

# What is the steady state of the transition matrix?
steady_state = np.linalg.solve(transition_matrix.T, np.ones(5))

print(f"The steady state of the transition matrix is {steady_state}")

# Absorbing state means that once the chain enters the state, it will never leave it

# Example: Moody's & S&P Transition Matriices are not direcly used much.
# Why? Eventral heap Map -> "D"

# Portfolio managers make their "own matrix"
# Modification 1: Shrink the time interval to (maybe) 1 month.
# Transition -> Birth - Death Process. Which means the transition matrix is no longer a stochastic matrix.

# Modifid Transition Matrix. (Modification 2): Active portfolio management in state BB
# [AAA, AA, A, BBB, BB, B, CCC, CC, C, D]
# AAA -> [0.9081, 0.0919, 0.0, 0, 0, 0, 0, 0, 0, 0]
# AA -> [0.007, 0.9065, 0.0865, 0, 0, 0, 0, 0, 0, 0]
# A -> [0, 0.03, 0.92, 0, 0, 0, 0, 0, 0, 0]
# BBB -> [0, 0, 0, 0.9, 0.08, 0.01, 0, 0, 0, 0]
# BB -> [0, 0, 0, 0.05, 0.87, 0.07, 0, 0, 0, 0]
# B -> [0, 0, 0, 0, 0.07, 0.93, 0, 0, 0, 0]
# CCC -> [0, 0, 0, 0, 0, 0, 0.9, 0.06, 0.03, 0]
# CC -> [0, 0, 0, 0, 0, 0, 0.05, 0.87, 0.07, 0]
# C -> [0, 0, 0, 0, 0, 0, 0, 0.07, 0.93, 0]
# D -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
