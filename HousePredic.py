import numpy as np
import pandas as pd

dataset = pd.read_csv("D:/Python/HeThongThoiGianThuc/BasicML/house-prices.csv")
dataset = dataset.replace("East", 0)
dataset = dataset.replace("West", 1)
dataset = dataset.replace("North", 2)

dataset = dataset.replace("No", 0)
dataset = dataset.replace("Yes", 1)

sqft = np.array([dataset["SqFt"].tolist()]).T
badrooms = np.array([dataset["Bedrooms"].tolist()]).T
bathrooms = np.array([dataset["Bathrooms"].tolist()]).T
offers = np.array([dataset["Offers"].tolist()]).T
brick = np.array([dataset["Brick"].tolist()]).T
neighborhood = np.array([dataset["Neighborhood"].tolist()]).T
prices = np.array([dataset["Price"].tolist()]).T

Y = np.array(prices).T
X = np.concatenate((sqft, badrooms, bathrooms, offers, brick, neighborhood), axis=1)

m = 127
x_train, x_test = X[:m,:], X[m:,:]
y_train, y_test = Y[:, :m], Y[:, m:]

Xbar = x_train

def grad(theta):
    yhat = (np.dot(x_train, theta) - y_train) 
    gradient = np.dot(x_train.T, yhat)
    print(gradient)
    return gradient/Xbar.shape[0]

def gradDecent(init_theta, learning_rate):
    theta_arr = [init_theta]
    for i in range(100):
        theta_new = theta_arr[-1] - learning_rate*grad(theta_arr[-1])
        if (np.linalg.norm(grad(theta_new)) < 1e-3):
            return theta_new
        theta_arr.append(theta_new)
    return theta_arr[-1]

theta = gradDecent([[2],[3],[5],[6],[7],[9]], 0.01)
print(theta)