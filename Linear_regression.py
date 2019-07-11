import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import numpy as np

data = pd.read_csv("D:/Python/HeThongThoiGianThuc/data.csv")
heights = data['Height'].tolist()
heights = np.asarray(heights)

weights = data['Weight'].tolist()
weights = np.asarray(weights)

index_test = [len(heights) - 1, len(heights) - 2]

heights_test = np.array([heights[len(heights) - 1], heights[len(heights) - 2]])
weights_test = np.array([weights[len(weights) - 1], weights[len(weights) - 2]])

heights_train = np.delete(heights, index_test).reshape(-1,1)
weights_train = np.delete(weights, index_test).reshape(-1,1)


linear = linear_model.LinearRegression()
linear.fit(heights_train, weights_train)

predict_weight = linear.predict(heights_test.reshape(-1,1))
print("predict weight is: {}".format(predict_weight))
print("Actual weight is: {}".format(weights_test))

plt.scatter(heights, weights)
plt.plot(heights.reshape(-1, 1), linear.predict(heights.reshape(-1, 1)), color='red', linewidth=3)
plt.show()