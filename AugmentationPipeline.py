import cv2
import os
import datetime
import numpy as np
from AugmentationClass import Augmentation
#   Run the AugmentationClass codes, so can refer to in this script

#   Pipeline for augmentation model testing
self = Augmentation("F:\\mphil\\Cheryl's folder\\go pro\\bug fixing")
#   directory of original frames

self.shear(0.5, 0.5, True, True, "left", "up", False)
self.export("F:\\mphil\\Cheryl's folder\\bug fixing")
self.next()
self.shear(0.5, 0.5, True, True, "right", "down", False)
self.export("F:\\mphil\\Cheryl's folder\\bug fixing")
self.next()
self.shear(0.5, 0.5, True, True, "left", "down", False)
self.export("F:\\mphil\\Cheryl's folder\\bug fixing")
self.next()
self.shear(0.5, 0.5, True, True, "right", "up", False)
self.export("F:\\mphil\\Cheryl's folder\\bug fixing")
self.next()