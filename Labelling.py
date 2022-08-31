import os, cv2
import numpy as np

os.chdir("D:\\Mirror\\mphil\\Cheryl's folder\\drone")

train = cv2.imread("DJI_0570.jpg")
cv2.imshow("Image", train); cv2.waitKey(0)

os.mkdir(".\\cropped")
os.chdir(".\\cropped")

dim = 400
for i in np.floor(np.linspace(0, train.shape[0], int(np.ceil(train.shape[0] / dim) + 1))).astype(int)[0:-1]:
    for j in np.floor(np.linspace(0, train.shape[1], int(np.ceil(train.shape[1] / dim) + 1))).astype(int)[0:-1]:
        cv2.imwrite(f"DJI_0570_{i}_{j}.jpg", train[i:i+dim, j:j+dim])
