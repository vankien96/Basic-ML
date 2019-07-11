import numpy as np
import tensorflow as tf


X = np.array([[1, 2, 3, 4, 5]]).T
Y = np.array([[6, 7, 4, 3, 2]]).T

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, y)
# Ainvert = np.linalg.pinv(A)
# w = np.dot(Ainvert, b)

def grad(theta):
    global X, Y
    gradient = np.dot(Xbar.T, (np.dot(Xbar, theta) - Y))
    return gradient/Xbar.shape[0]

theta_arr = [[[4],[3]]]

def gradDecent(init_theta, learning_rate):
    theta_arr = [init_theta]
    for i in range(10000000):
        theta_new = theta_arr[-1] - learning_rate*grad(theta_arr[-1])
        if (np.linalg.norm(grad(theta_new)) < 1e-3):
            print(i)
            return theta_new
        theta_arr.append(theta_new)
    return theta_arr[-1]

def gradientDescentMomentum(init_theta, learning_rate, gamma):
    theta_arr = [init_theta]
    v_old = np.zeros_like(init_theta)
    for i in range(10000000):
        v_new = gamma*v_old + learning_rate*grad(theta_arr[-1])
        theta_new = theta_arr[-1] - v_new
        if (np.linalg.norm(grad(theta_new)) < 1e-3):
            print(i)
            return theta_new
        theta_arr.append(theta_new)
        v_old = v_new
    return theta_arr[-1]

def NAG(init_theta, learning_rate, gamma):
    theta_arr = [init_theta]
    v_old = np.zeros_like(init_theta)
    for i in range(10000000):
        v_new = gamma*v_old + learning_rate*grad(theta_arr[-1] - gamma*v_old)
        theta_new = theta_arr[-1] - v_new
        if (np.linalg.norm(grad(theta_new)) < 1e-3):
            print(i)
            return theta_new
        theta_arr.append(theta_new)
        v_old = v_new
    return theta_arr[-1]

print(gradDecent([[4],[3]], 0.01))
print(gradientDescentMomentum([[4],[3]], 0.01, 0.9))
print(NAG([[4],[3]], 0.01, 0.9))

    