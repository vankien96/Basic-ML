import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict

mnist = fetch_mldata(dataname='mnist-original', data_name='D:/Python/HeThongThoiGianThuc/mldata')

X, y = mnist["data"], mnist["target"]
X = X / 255

# lay ra tat ca chu so 0
# x0 = x_all[np.where(y_all==0)[0]]
# x1 = x_all[np.where(y_all==1)[0]]
# y0 = np.zeros(x0.shape[0])
# y1 = np.ones(x1.shape[0])

# X = np.concatenate((x0, x1), axis = 0)
# Y = np.concatenate((y0, y1))

# # print(x0.shape)
# # plt.scatter(x_all, y_all)
# # plt.show()
# m = 4000
# m_train = X.shape[0] - m
# X_train, X_test = X[:m_train].T, X[m_train:].T
# Y_train, Y_test = Y[:m_train].T, Y[m_train:].T
# # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=4000)

# print(x_train)
# print(y_train)

digits = 2

x0 = X[np.where(y==0)[0]]
x1 = X[np.where(y==1)[0]]
y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

X = np.concatenate((x0, x1), axis = 0)
y = np.concatenate((y0, y1))

examples = y.shape[0]
y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

m = 13000
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n_x = X_train.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))

X = X_train
Y = Y_train

for i in range(100):

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = compute_multiclass_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

##################################################################
def is_inside(rect1, rect2):
    if rect1[0] > rect2[0] and (rect1[0] + rect1[2]) < (rect2[0] + rect2[2]) and rect1[1] > rect2[1] and (rect1[1] + rect1[3]) < (rect2[1] + rect2[3]):
        return True
    else:
        return False

def check_contain_another_rect(rect, rects):
    for item in rects:
        if is_inside(rect, item):
            return True
    return False

image = cv2.imread("D:/Python/HeThongThoiGianThuc/IMG.JPG")
height, width = image.shape[:2]
scaledWidth = 900
scaledHeight = int((scaledWidth * height) / width)
image = cv2.resize(image, (scaledWidth, scaledHeight), fx= 0.5, fy=0.5, interpolation= cv2.INTER_AREA)
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)

kernel = np.ones((3,3),dtype='uint8')
ret, im_th = cv2.threshold(im_gray, 115, 255, cv2.THRESH_BINARY_INV)
canny = cv2.Canny(im_th, 70, 170)
# canny = cv2.dilate(canny, kernel, iterations = 1)
_, ctrs, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

new_rects = []
for rect in rects:
    if rect[2] > 10 and rect[3] > 20 and rect[3] < 100 and not check_contain_another_rect(rect, rects):
        new_rects.append(rect)

# model = joblib.load("D:/Python/HeThongThoiGianThuc/mldata/digital.pkl")

new_rects = np.array(new_rects)
data_input = OrderedDict()
for rect in new_rects:
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3]/2 - leng/2)
    pt2 = int(rect[0] + rect[2]/2 - leng/2)
    roi = im_th[pt1: pt1 + leng, pt2: pt2+leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    number = np.array([roi]).reshape(1, (28*28))
    data_input[(pt2,pt1, pt2 + leng, pt1+leng)] = number

for point, number in data_input.items():
    print(number.reshape(-1,1))
    Z1 = np.matmul(W1, number.reshape(-1,1)) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    predictions = np.argmax(A2, axis=0)
    cv2.rectangle(im_gray, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 1)
    cv2.putText(im_gray, str(int(predictions)), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Image", im_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()