import cv2
import os
import datetime
import numpy as np
from AugmentationClass import Augmentation
self = Augmentation("F:\\mphil\\Cheryl's folder\\go pro\\Stanley_2")

for _ in range(3):
    self.rotate(180, "both", True)
    self.brightness(0.3, "both", True)
    self.colour(0.1, "both", ("b", "g", "r"), True)
    self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing3")
    self.next()

self.rotate(30, "both", True)
self.export("F:\\mphil\\Cheryl's folder\\testing2")
self.next()
self.flip(True, True, True)
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()
self.saturation(0.3, "both", True)
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()
self.brightness(0.3, "both", True)
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()
self.colour(0.1, "both", ("b", "g", "r"), True)
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()
self.blur(10, True)
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()
self.sharpen()
self.export("D:\\Mirror\\mphil\\Cheryl's folder\\testing2")
self.next()

a = self.aug_img[10]

import cv2
cv2.imshow("", a); cv2.waitKey()
M = cv2.getRotationMatrix2D((a.shape[1] // 2, a.shape[0] // 2), 30, 1)
b = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
cv2.imshow("", b); cv2.waitKey()

cv2.transform()