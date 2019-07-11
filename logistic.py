import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import cv2

mnist = fetch_mldata(dataname='mnist-original', data_name='D:/Python/HeThongThoiGianThuc/mldata')

x_all = mnist.data
y_all = mnist.target

# lay ra tat ca chu so 0
x0 = x_all[np.where(y_all==0)[0]]
x1 = x_all[np.where(y_all==1)[0]]
y0 = np.zeros(x0.shape[0])
y1 = np.ones(x1.shape[0])

X = np.concatenate((x0, x1), axis = 0)
Y = np.concatenate((y0, y1))

# print(x0.shape)
# plt.scatter(x_all, y_all)
# plt.show()
m = 13000
m_train = X.shape[0] - m
x_train, x_test = X[:m_train], X[m_train:]
y_train, y_test = Y[:m_train], Y[m_train:]
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=4000)

print(x_train)
print(y_train)

# i = 3
# plt.imshow(x0_train[:,i].reshape(28,28))
# plt.axis("off")
# plt.show()

# model = LogisticRegression(C=1e5)
# model.fit(x_all, y_all)

# joblib.dump(model, "D:/Python/HeThongThoiGianThuc/mldata/digital.pkl", compress=3)

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

model = joblib.load("D:/Python/HeThongThoiGianThuc/mldata/digital.pkl")

new_rects = np.array(new_rects)

for rect in new_rects:
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3]/2 - leng/2)
    pt2 = int(rect[0] + rect[2]/2 - leng/2)
    roi = im_th[pt1: pt1 + leng, pt2: pt2+leng]
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    number = np.array([roi]).reshape(1, (28*28))
    ret = model.predict(number)
    cv2.rectangle(im_gray, (pt2, pt1), (pt2 + leng, pt1+leng), (0, 255, 0), 1)
    cv2.putText(im_gray, str(int(ret[0])), (pt2, pt1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Image", im_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


