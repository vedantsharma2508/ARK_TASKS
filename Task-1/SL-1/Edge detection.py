import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 

img = cv.imread('table.png',1)
canny = cv.Canny(img,88,125)

titles = ['img',"canny"]
images = [img,canny]

plt.figure(figsize=(10, 5))

for i in range (2) :
    plt.subplot(1,2,i+1) , plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])

plt.show()

# Hough transform

lines = cv.HoughLines(canny,1,np.pi/180,200)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv.imshow('Hough Lines', img)
cv.waitKey(0)
cv.destroyAllWindows()
