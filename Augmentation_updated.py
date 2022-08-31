import cv2
import os
import copy
import numpy as np

class Augmentation:
    def __init__(self, directory: str, batch_start: int = 1):
        # directory: folder that contain images and bounding boxes, as well as a txt file describing object classes
        # batch_start: new batch without overwriting old one in same directory

        self.directory = directory
        #  where the images and bounding box txts are located
        self.file_list = os.listdir(directory)
        #  get the list in all files in directory
        fnames = [f.split(".")[0] for f in self.file_list]
        #  remove filename extensions
        with open(f"{self.directory}/classes.txt", "r") as classes:  # Open the object class file (refer to as classes)
            self.classes = classes.read()
            #  extract what is in the file
        self.annotated = list(set([f for f in fnames if fnames.count(f) > 1]))
        #  get name if there are duplicated file names (f) (i.e. hv bounding box file; annotated)
        self.annotated.sort(key=lambda x: int(x.split("_")[-1]))
        #  arrange the annotated file names in ascending order
        self.n = len(self.annotated)
        #  the number of annotated files

        print("Loading images...")  # progress indicator
        self.img = [cv2.imread(f"{self.directory}/{f}.jpg") for f in self.annotated]
        #  read images with cv2

        print("Loading bounding boxes...")  # progress indicator
        self.bbox = []
        #  initialize list to store bounding boxes
        for f in self.annotated:
            with open(f"{self.directory}/{f}.txt", "r") as txt:  # refer file to txt
                self.bbox.append(
                    np.array([line.split(" ") for line in txt.read().split("\n")[:-1]]).astype(float)
                )  # read bounding boxes and put each bounding box (as one array) each by each into the list (self.bbox)

        self.aug_img = copy.deepcopy(self.img)
        #  create a copy of images to be augmented
        self.aug_bbox = copy.deepcopy(self.bbox)
        #  create a copy of bounding boxes to be augmented

        self.batch = f"A{batch_start}"
        #   save which batch this is and to check what augmentations have been done -> for file naming afterwards

        self.aug_steps = {self.batch: []}  # list of augmentations made for each batch

    def saturation(self, per_change: float = 0.3, direction: str = "both", random: bool = True):
        #   per_change = range of max. percentage_change
        #   direction = can + or - ("up"/"down"/"both")
        #   if random, actual per_change = from 0 to user input number of per_change;
        #   else actual per_change = user input number of per_change

        assert direction in ["both", "up", "down"]
        #   control user input for direction to be "both", "up", "down" only

        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
            #  to control user input to be between 0 and 1
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")
            #   to control user input, if "both" must be random

        low = -per_change if direction != "up" else 0
        #   If direction is both or down, low = -per_change, if up, low = 0
        high = per_change if direction != "down" else 0  # If direction is both or up
        #   if direction is both or up, high = per_change, if down, high = 0
        #   both = -per_change to per_change; up = 0 to per_change; down = -per_change to 0
        #   Controlling the Max. and Min. boundary of percentage change in randomness

        if random:
            correction = np.random.uniform(low, high, self.n)
            #   draw samples from a uniform distribution randomly from the number of samples
            #   and return a new array of random numbers within per_change range
        else:
            correction = np.full(self.n, per_change) if direction == "up" else np.full(self.n, -per_change)
            #   return per_change or -per_change based on the direction

        for r, (n, i) in zip(correction, enumerate(self.aug_img)):
            #  r = correction = corrected saturation
            #  n = index
            #  i = image

            i = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
            #   change RGB to HSV format
            if r > 0:
                i[:, :, 1] = np.where(i[:, :, 1].astype(float) + (r * 256) > 255, 255,
                                      i[:, :, 1] + (r * 256))
                #  Image format: height:width:HSV layer (1 = saturation),
                #  if r > 255 (cannot be), set to 255, else augmented number
            else:
                i[:, :, 1] = np.where(i[:, :, 1].astype(float) + (r * 256) < 0, 0, i[:, :, 1] + (r * 256))
                #  r cannot be lower than 0
            self.aug_img[n] = cv2.cvtColor(i, cv2.COLOR_HSV2BGR)
            #  change back to BGR
        self.aug_steps[self.batch].append(f"saturation: per_change={per_change} direction={direction} random={random}")
        # remark what augmentation has been done to the images

    def brightness(self, per_change: float = 0.3, direction: str = "both", random: bool = True):
        #   per_change = range of max. percentage_change
        #   direction = can + or - ("up"/"down"/"both")
        #   if random, actual per_change = from 0 to user input number of per_change;
        #   else actual per_change = user input number of per_change

        assert direction in ["both", "up", "down"]
        #   control user input for direction to be "both", "up", "down" only

        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
            #  to control user input to be between 0 and 1
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")
            #   to control user input, if "both" must be random

        low = -per_change if direction != "up" else 0
        #   If direction is both or down, low = -per_change, if up, low = 0
        high = per_change if direction != "down" else 0
        #   if direction is both or up, high = per_change, if down, high = 0
        #   both = -per_change to per_change; up = 0 to per_change; down = -per_change to 0
        #   Controlling the Max. and Min. boundary of percentage change in randomness

        if random:
            correction = np.random.uniform(low, high, self.n)
            #   draw samples from a uniform distribution randomly from the number of samples
            #   and return a new array of random numbers within per_change range
        else:
            correction = np.full(self.n, per_change) if direction == "up" else np.full(self.n, -per_change)
            #   return per_change or -per_change based on the direction

        for r, (n, i) in zip(correction, enumerate(self.aug_img)):
            #  r = correction = corrected brightness
            #  n = index
            #  i = image

            i = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
            #   change RGB to HSV format
            if r > 0:
                i[:, :, 2] = np.where(i[:, :, 2].astype(float) + (r * 256) > 255, 255, i[:, :, 2] + (r * 256))
                #  Image format: height:width:HSV layer (2 = brightness),
                #  if r > 255 (cannot be), set to 255, else augmented number
            else:
                i[:, :, 2] = np.where(i[:, :, 2].astype(float) + (r * 256) < 0, 0, i[:, :, 2] + (r * 256))
                #  r cannot be lower than 0
            self.aug_img[n] = cv2.cvtColor(i, cv2.COLOR_HSV2BGR)
            #  change back to BGR
        self.aug_steps[self.batch].append(f"brightness: per_change={per_change} direction={direction} random={random}")
        # remark what augmentation has been done to the images

    def colour(self, per_change: float = 0.1, direction: str = "both",
               bgr: tuple = ("b", "g", "r"), random: bool = True):
        #   per_change = range of max. percentage_change
        #   direction = can + or - ("up"/"down"/"both")
        #   bgr = BGR (Blue, Green, Red)
        #   if random, actual per_change = from 0 to user input number of per_change;
        #   else actual per_change = user input number of per_change

        assert direction in ["both", "up", "down"]
        #   control user input for direction to be "both", "up", "down" only

        assert 0 < len(bgr) < 4
        #   control user input length to not exceed 3 inputs

        assert all([c in ("b", "g", "r") for c in set(bgr)])
        #   control user input to only hv "b" "g" and "r"
        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
            #  to control user input to be between 0 and 1
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")
            #   to control user input, if "both" must be random

        low = -per_change if direction != "up" else 0
        #   If direction is both or down, low = -per_change, if up, low = 0
        high = per_change if direction != "down" else 0
        #   if direction is both or up, high = per_change, if down, high = 0
        #   both = -per_change to per_change; up = 0 to per_change; down = -per_change to 0
        #   Controlling the Max. and Min. boundary of percentage change in randomness

        key = {"b": 0, "g": 1, "r": 2}
        #   set a fixed key for b, g, r to refer to their respective layers

        for n, i in enumerate(self.aug_img):
            #  n = index
            #  i = image

            if random:
                correction = np.random.uniform(low, high, len(bgr))
                #   draw samples from a uniform distribution randomly from the number of samples
                #   and return a new array of random numbers within per_change range
            else:
                correction = np.full(len(bgr), per_change) if direction == "up" else np.full(len(bgr), -per_change)
                #   return per_change or -per_change based on the direction
            for (nc, c), r in zip(enumerate(bgr), correction):
                #  nc = index of colour
                #  c = colour = layer
                #  r = correction = corrected colour
                if r > 0:
                    i[:, :, key[c]] = np.where(i[:, :, key[c]].astype(float) + (r * 256) > 255, 255,
                                               i[:, :, key[c]] + (r * 256))
                    # Image format:  height:width:BGR layer
                    #  if r > 255 (cannot be), set to 255, else augmented number
                else:
                    i[:, :, key[c]] = np.where(i[:, :, key[c]].astype(float) + (r * 256) < 0, 0,
                                               i[:, :, key[c]] + (r * 256))
                    #  r cannot be lower than 0
            self.aug_img[n] = i
            #  save augmentation
        self.aug_steps[self.batch].append(f"color: per_change={per_change} direction={direction} "
                                          f"bgr={bgr} random={random}")
        # remark what augmentation has been done to the images

    def blur(self, max_ksize: int = 10, random: bool = True):
        #   max_ksize = kernel size, kernel = a convolution averaging filter (n X m pixels)
        #   if random, actual ksize = from 0 to user input number of max_ksize;
        #   else actual ksize = user input number of max_ksize

        if max_ksize < 0:
            raise ValueError("max_ksize cannot be negative")
        #   control user input for direction to be > 0


        ksize = np.round(np.random.uniform(1, max_ksize, self.n)).astype(int) if random else np.full(self.n, max_ksize)
        #   draw samples from a uniform distribution randomly from the number of samples
        #   and return a new array of random numbers within max_ksize range (rounded off)
        self.aug_img = [cv2.blur(i, (k, k)) for k, i in zip(ksize, self.aug_img)]
        #   blur image
        self.aug_steps[self.batch].append(f"blur: max_ksize={max_ksize} random={random}")
        # remark what augmentation has been done to the images

    def sharpen(self):
        self.aug_img = [cv2.filter2D(i, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])) for i in self.aug_img]
        #   sharpen image
        #   new input for centre pixel = number in matrix multiply the respective pixel then summing up
        self.aug_steps[self.batch].append(f"sharpen")
        # remark what augmentation has been done to the images

    def flip(self, horizontal: bool = True, vertical: bool = True, random: bool = True):
        #   horizontal, vertical = False -> don't flip
        #   Random = flip or not flip (if true)

        flipx = np.round(np.random.uniform(0, horizontal, self.n)) if random else np.full(self.n, horizontal)
        flipy = np.round(np.random.uniform(0, vertical, self.n)) if random else np.full(self.n, vertical)
        #   draw samples from [0, 1] randomly return a new array of 0s and 1s

        for x, y, b, (n, i) in zip(flipx, flipy, self.aug_bbox, enumerate(self.aug_img)):
            #  x = flip horizontally?
            #  y = flip vertically?
            #  b = bounding boxes
            #  n = index
            #  i = image
            self.aug_img[n] = cv2.flip(i, -1) if x and y else cv2.flip(i, 1) if x else cv2.flip(i, 0) if y else i
            #   flip image in two directions based on x and y
            #   i = no flipping at all
            if x: b[:, 1] = 1 - b[:, 1]
            if y: b[:, 2] = 1 - b[:, 2]
            #   have to flip bounding boxes accordingly
            #   [yolo bbox format = (class, x (in relative to whole img, %), y, w, h)]
            self.aug_bbox[n] = b
            #  save augmentation
        self.aug_steps[self.batch].append(f"flip: horizontal={horizontal} vertical={vertical} random={random}")
        # remark what augmentation has been done to the images

    def rotate(self, max_rotation: float = 180, direction: str = "both", random: bool = True):
        #   max_rotation = range of max. rotation angle
        #   direction = can + or - ("clockwise"/"anti-clockwise"/"both")
        #   if random, actual rotation = from 0 to user input number of max_rotation;
        #   else actual rotation = user input number of max_rotation

        assert direction in ["both", "clockwise", "anti-clockwise"]
        #   control user input for direction to be "both", "clockwise", "anti-clockwise" only

        if max_rotation > 180 or max_rotation < 0:
            raise ValueError("Invalid rotation angle")
            #  to control user input to be between 0 and 180
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")
            #   to control user input, if "both" must be random

        low = -max_rotation if direction != "clockwise" else 0
        #   = both or anti-clockwise
        high = max_rotation if direction != "anti-clockwise" else 0
        #   = both or clockwise
        if random:
            correction = np.random.uniform(low, high, self.n)
            #   draw samples from a uniform distribution randomly from the number of samples
            #   and return a new array of random numbers within per_change range
        else:
            correction = np.full(self.n, max_rotation) if direction == "clockwise" else np.full(self.n, -max_rotation)
            #   return max_rotation or -max_rotation based on the direction ##########

        for r, b, (n, i) in zip(correction, self.aug_bbox, enumerate(self.aug_img)):
            M = cv2.getRotationMatrix2D((i.shape[1] // 2, i.shape[0] // 2), r, 1) # changed
            abs_cos = np.abs(M[0, 0]) # changed
            abs_sin = np.abs(M[0, 1]) # changed
            bound_w = int(i.shape[0] * abs_sin + i.shape[1] * abs_cos) # changed
            bound_h = int(i.shape[0] * abs_cos + i.shape[1] * abs_sin) # changed
            M[0, 2] += bound_w / 2 - i.shape[1] / 2 # changed
            M[1, 2] += bound_h / 2 - i.shape[0] / 2 # changed
            #   creating rotation matrix
            #   cv2.getRotationMatrix2D = (rotation centre, rotation angle, scaling ratio)
            #   i.shape[1] // 2, i.shape[0] // 2 = centre of image
            #   (np.min = return the min. value among compared)
            #   = scale down the photo to avoid edges of image being chop off when rotating
            self.aug_img[n] = cv2.warpAffine(i, M, (bound_w, bound_h)) # changed
            #   applying matrix with photo scaling down
            #   (cv2.warpAffine() = (img, matrix, (w,h) =
            #   (dimension of output image) = (i.shape[1], i.shape[0]) = same as original img)

            coords = [np.array([[(p[1] - p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] - p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1]]) for p in b]
            #   to get the coordinates (in absolute value) of the four corners
            #   p = each bounding box in each bbox file (b = bbox files)
            #   [0] = class, [1] = x (relative) , [2] = y (relative), [3] = width, [4] = height of each bounding box
            #   i.shape[1] = width, i.shape[0] = height of IMG
            #   (x,y) of bbox = (x.mean - (w/2)), (y.mean - (h/2))
            #   (x,y) * img width = absolute value of the bbox

            new_coords = [M.dot(p.T).T for p in coords]
            #  to find where are the new coordinates of the bbox in the new rotated image
            #   .dot() = dot product = matrix (M) * matrix (p)
            #   (np.array.T = transpose of the matrix), i.e. 4x2 to 2x4
            #   M.dot(p.T).T = transpose for cal. then transpose back to ori.
            self.aug_bbox[n] = np.array([np.array([o[0],
                                                   (p[:, 0].max() + p[:, 0].min()) / 2 / bound_w,
                                                   (p[:, 1].max() + p[:, 1].min()) / 2 / bound_h,
                                                   (p[:, 0].max() - p[:, 0].min()) / bound_w,
                                                   (p[:, 1].max() - p[:, 1].min()) / bound_h]) for o, p in
                                         zip(b, new_coords)])  # changed
            #   create new bbox that can enclose old bbox and in same orientation with the "frame" and in Yolo format
            #   o = b = o[0] = class
            #   p = new coordinates, [0] = x, [1] = y
            #   i.e. (p[:, 0].max() + p[:, 0].min()) / 2 / i.shape[1] = xmin + xmin
        self.aug_steps[self.batch].append(f"rotate: max_rotation={max_rotation} direction={direction} random={random}")

    def shear(self, max_shear: float = 0.5, horizontal: bool = True, vertical: bool = True,
              direction: str = "both", random: bool = True):
        #   horizontal = x-axis
        #   vertical = y-axis
        #   If random, img can: no shear at all, shear both horizontal and vertical, shear along either
        #   in random range within max_shear and in random direction
        assert direction in ["both", "up", "down"]
        #   [direction = up or down, or right or left (for horizontal shear) = think new term]
        if max_shear > 1 or max_shear < 0:
            raise ValueError("Invalid rotation angle")
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")
        if random:
            correctionx = np.random.uniform(0, max_shear, self.n) if horizontal else np.zeros(self.n)
            #   (np.zero() = Return a new array of given shape and type, filled with zeros)
            #   correctrionx = random numbers within max_shear range when horizontal shear,
            #   else zero as not shearing along x-asix
            correctiony = np.random.uniform(0, max_shear, self.n) if vertical else np.zeros(self.n)

            flipx = np.zeros(self.n).astype(bool) if not horizontal or direction == "up" else np.ones(self.n).astype(
                bool) if direction == "down" else np.random.uniform(0, 1, self.n).round().astype(bool)
            #   np.zero().astype(bool) = 0 = False
            #   If not horizontal = when horizontal = F, wont shear along x-asix, so no need filip along x
            #   If direction = "up" = vertical shear up = accords to original direction = no need flip
            #   If direction = "down" = vertical shear down = need flip
            #   If direction = "both" = either shear up or down so determined by flipx eiter T (1) or F (0)
            flipy = np.zeros(self.n).astype(bool) if not vertical or direction == "up" else np.ones(self.n).astype(
                bool) if direction == "down" else np.random.uniform(0, 1, self.n).round().astype(bool)
            #   when vertical = F
        else:
            correctionx = np.full(self.n, max_shear) if horizontal else np.zeros(self.n)
            #   np.full() = Return a new array of given shape and type, filled with fill_value (i.e. max_shear value)
            correctiony = np.full(self.n, max_shear) if vertical else np.zeros(self.n)
            flipx = np.zeros(self.n).astype(bool) if not horizontal or direction == "up" else np.ones(self.n).astype(
                bool)
            flipy = np.zeros(self.n).astype(bool) if not vertical or direction == "up" else np.ones(self.n).astype(bool)

        for cx, cy, fx, fy, b, (n, i) in zip(correctionx, correctiony, flipx, flipy, self.aug_bbox,
                                             enumerate(self.aug_img)):
            i = cv2.flip(i, -1) if fx and fy else cv2.flip(i, 1) if fx else cv2.flip(i, 0) if fy else i
            #   cv2.flip(i, -1), -1 = flipping along both axis, 1 = y, 0 = x
            if fx: b[:, 1] = 1 - b[:, 1]
            #   if fx =  True = need to change (i.e. flip) the [1] = x-location of bbox
            if fy: b[:, 2] = 1 - b[:, 2]
            coords = [np.array([[(p[1] - p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] - p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1]]) for p in b]
            #   for p in b = for each element (p) in bbox array (b), make changes and = coords
            #   [0] = class, [1] = x (relative) , [2] = y (relative), [3] = width, [4] = height of each bounding box
            #   i.shape[1] = width, i.shape[0] = height of IMG
            bounds = np.array([np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]) for p in coords])
            #   for p in coords = for each element (p) in coords, make changes and = bounds
            #   get boundary of bounding box
            #   p[:, 0].min(), p[:, 1].min() = xmin,ymin (coords of the angle closest to origin)
            #   [p[:, 0].min(), p[:, 1].min() = xmax, ymax (coords of the angle furthest to origin)
            i = cv2.warpPerspective(i, np.array([[1, cx, 0],
                                                 [cy, 1, 0],
                                                 [0, 0, 1]]),
                                    (int(i.shape[1] + i.shape[0] * cx), int(i.shape[0] + i.shape[1] * cy)))
            #   cv2.warpPerspective = shear the image (img, shearing matrix, dimensino of output image)

            bounds[:, [0, 2]] += ((bounds[:, [1, 3]]) * cx).astype(int)
            bounds[:, [1, 3]] += ((bounds[:, [0, 2]]) * cy).astype(int)
            #   [0] = xmin, [1] = ymin, [2] = xmax, [3] = ymax
            #   bounds[:, [0, 2]] = (:) = for all elements in array bounds, xmin and xmax
            #   +=: a = 1, a += 1, output a = 2
            #   shear bbox and creating a frame that can enclose new bbox

            b = np.array([np.array([int(o[0]),
                                    (p[0] + p[2]) / 2 / i.shape[1],
                                    (p[1] + p[3]) / 2 / i.shape[0],
                                    (p[2] - p[0]) / i.shape[1],
                                    (p[3] - p[1]) / i.shape[0]]) for o, p in zip(b, bounds)])
            #   change back to yolo format

            self.aug_img[n] = cv2.flip(i, -1) if fx and fy else cv2.flip(i, 1) if fx else cv2.flip(i, 0) if fy else i
            if fx: b[:, 1] = 1 - b[:, 1]
            if fy: b[:, 2] = 1 - b[:, 2]
            self.aug_bbox[n] = b
        self.aug_steps[self.batch].append(f"shear: max_shear={max_shear} horizontal={horizontal} vertical={vertical} "
                                          f"direction={direction} random={random}")

    def export(self, destination: str = ""):
        for name, i in zip(self.annotated, self.aug_img):
            cv2.imwrite(f"{destination}/{name}_{self.batch}.jpg", i)
            #   name and save augmented images as (?)
            #   (cv2.imwrite () = save an image (with given name) to any storage device)
            #   self.annotated = list of names of img with bbox
            #   save in {destination}, in file format of: {name}_batch number.jpg
        for name, b in zip(self.annotated, self.aug_bbox):
            #   name and save augmented bbox file as
            with open(f"{destination}/{name}_{self.batch}.txt", "w") as coord:
                coord.write("\n".join([" ".join([str(e) if e != 0 else str(int(e)) for e in l]) for l in b.tolist()]) + "\n")
                #   create .txt format
                #   a = ['1','2'], " ".join(a) = 1 2
        with open(f"{destination}/classes.txt", "w") as classes:
            #  name and save class file as
            classes.write(self.classes)
            #  create class information
        try:
            with open(f"{destination}/augmentation steps.txt", "r") as log:
                old_log = log.read()
                #  read the augmentation steps.txt file
            old_log = old_log.split("\n")[:-1]
            #  split each line (each batch info) into one element
            keys = [l.split(" -- ")[0] for l in old_log]
            #   make a list of keys, (batch)
            items = [l.split(" -- ")[1].split("; ") for l in old_log]
            #   make a list of items, (steps of augmentation) (l.split(" -- ")[1])
            #   and a sub-list of each step (.split("; "))
            for k, i in zip(keys, items):
                if k not in self.aug_steps.keys():
                    self.aug_steps[k] = i
                    #   If keys is not in new log, add back in for each key and respective item
                    #   Purpose: renew the log and save files in same directory when making new batches
            self.aug_steps = {k: self.aug_steps[k] for k in sorted(self.aug_steps, key=lambda x: int(x.split("A")[-1]))}
            #   reorder the keys into order
        except FileNotFoundError:
            pass
        #   FileNotFoundError = have not done anything yet so no old files = pass (do not do this step)
        #   try:... except = if no exception occurs through out the run of the codes,
        #   except clause is skipped and execution of the code is finished ,
        #   if there is an exception, (Error that matches with stated), rest of code skipped and execute except clause
        with open(f"{destination}/augmentation steps.txt", "w") as log:
            log.write("\n".join([key + " -- " + "; ".join(item) for (key, item) in self.aug_steps.items()]) + "\n")
            #    create .txt format

    def next(self):
        #   next() = create next batch (i.e. try next augmenting model)
        self.aug_img = copy.deepcopy(self.img)
        self.aug_bbox = copy.deepcopy(self.bbox)
        self.batch = f"A{int(self.batch[1:]) + 1}"
        self.aug_steps[self.batch] = []

    def reset(self):
        #   same as next() but previous things will be deleted unless created a new directory
        self.aug_img = copy.deepcopy(self.img)
        self.aug_bbox = copy.deepcopy(self.bbox)
        self.aug_steps[self.batch] = []

