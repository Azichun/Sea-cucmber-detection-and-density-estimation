import os, cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir("D:\\Mirror\\mphil\\Francis\\drone")

img1 = cv2.cvtColor(cv2.imread("dji_0071.jpg"), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("dji_0070.jpg"), cv2.COLOR_BGR2GRAY)

cv2.imshow("img1", img1); cv2.waitKey(0)
cv2.imshow("img2", img2); cv2.waitKey(0)

np.mean(img1 - img2)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
img1_min = cv2.erode(img1, kernel)
img1_max = cv2.dilate(img1, kernel)
img2_min = cv2.erode(img2, kernel)
img2_max = cv2.dilate(img2, kernel)

min_diff = [np.mean(img1_min[0:img1_min.shape[0] - i, :] - img2_min[i:img2_min.shape[0], :]) for i in np.arange(0, img1_min.shape[0], 10)]
max_diff = [np.mean(img1_max[0:img1_max.shape[0] - i, :] - img2_max[i:img2_max.shape[0], :]) for i in np.arange(0, img1_max.shape[0], 10)]

fig, ax = plt.subplots()
ax.plot(np.arange(0, img1_min.shape[0], 10), min_diff)
ax.plot(np.arange(0, img1_max.shape[0], 10), max_diff)