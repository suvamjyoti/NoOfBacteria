import numpy as np
import cv2
import random

# image source
img = cv2.imread('1.jpg')
# converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresholding
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

# morphological filtering for noise
# set kernel until you get best segmentation in the dilation result
kernel = np.ones((4, 4), np.uint8)
kernel1 = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
dilation = cv2.dilate(opening, kernel1, iterations=1)

# Marker labelling
retw, markers = cv2.connectedComponents(dilation)

# finding the size of image
w, h = markers.shape[::-1]
k = 0
font = cv2.FONT_HERSHEY_COMPLEX
for a in range(h):
    for b in range(w):
        if markers[a][b] > k:
            k = markers[a][b]
            cv2.putText(img, str(k), (b, a), font, 0.5, (0, 0, 0), 1)

f = 'total no of bacteria present ='
cv2.putText(img, f+str(k), (10, 20), font, 0.7, (0, 0, 0), 2)
print(k)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', dilation)
cv2.waitKey(0)
cv2.imwrite('result.jpg', img)
cv2.destroyAllWindows()