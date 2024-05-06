import cv2
import os
import matplotlib.pyplot as plt


left = cv2.imread('left.png',1)
right =  cv2.imread('right.img',1)

img_height = 360
img_width = 640

titles = ["left","right"]
images = [left,right]

plt.figure(figsize=(10, 5))

for i in range (2) :
    plt.subplot(1,2,i+1) , plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])

plt.show()