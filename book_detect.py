import numpy as np
import cv2

# image download, change color to gray, write new image
image = cv2.imread("images/example.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imwrite("images/gray.jpg", gray)

# recognize conture 
edged = cv2.Canny(gray, 10, 250)
cv2.imwrite("images/edged.jpg", edged)

# apply `closed` method
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("images/closed.jpg", closed)

# find conturse in image
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
total = 0

# conture cicle 
for c in cnts:
    # conture approximation
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #if conture have 4 top....
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1
# show count and write result image
print("{0} books on this photo".format(total))
cv2.imwrite("images/output.jpg", image)