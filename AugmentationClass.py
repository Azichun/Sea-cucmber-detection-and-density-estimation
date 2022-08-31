import cv2
import os
import copy
import numpy as np

class Augmentation:
    def __init__(self, directory: str, batch_start: int = 1):
        self.directory = directory  # directory (i.e. where the images and bounding box txts are located)
        self.file_list = os.listdir(directory)  # files in the directory
        fnames = [f.split(".")[0] for f in self.file_list]  # remove filename extensions
        with open(f"{self.directory}\\classes.txt", "r") as classes:
            self.classes = classes.read()
        self.annotated = list(set([f for f in fnames if fnames.count(f) > 1]))  # annotated images (only annotated images have bounding box txts)
        self.annotated.sort(key=lambda x: int(x.split("_")[-1]))  # sort by number
        self.n = len(self.annotated)

        print("Loading images...")  # progress indicator
        self.img = [cv2.imread(f"{self.directory}\\{f}.jpg") for f in
                                   self.annotated]  # read images with cv2

        print("Loading bounding boxes...")  # progress indicator
        self.bbox = []  # initialize list to store bounding boxes
        for f in self.annotated:
            with open(f"{self.directory}\\{f}.txt", "r") as txt:
                self.bbox.append(
                    np.array([line.split(" ") for line in txt.read().split("\n")[:-1]]).astype(float)
                )  # read bounding boxes
        self.aug_img = copy.deepcopy(self.img)
        self.aug_bbox = copy.deepcopy(self.bbox)
        self.batch = f"A{batch_start}"
        self.aug_steps = {self.batch: []}

    def saturation(self, per_change: float = 0.3, direction: str = "both", random: bool = True):
        assert direction in ["both", "up", "down"]
        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")

        low = -per_change if direction != "up" else 0
        high = per_change if direction != "down" else 0
        if random:
            correction = np.random.uniform(low, high, self.n)
        else:
            correction = np.full(self.n, per_change) if direction == "up" else np.full(self.n, -per_change)

        for r, (n, i) in zip(correction, enumerate(self.aug_img)):
            i = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
            if r > 0:
                i[:, :, 1] = np.where(i[:, :, 1].astype(float) + (r * 256) > 255, 255, i[:, :, 1] + (r * 256))
            else:
                i[:, :, 1] = np.where(i[:, :, 1].astype(float) + (r * 256) < 0, 0, i[:, :, 1] + (r * 256))
            self.aug_img[n] = cv2.cvtColor(i, cv2.COLOR_HSV2BGR)
        self.aug_steps[self.batch].append(f"saturation: per_change={per_change} direction={direction} random={random}")

    def brightness(self, per_change: float = 0.3, direction: str = "both", random: bool = True):
        assert direction in ["both", "up", "down"]
        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")

        low = -per_change if direction != "up" else 0
        high = per_change if direction != "down" else 0
        if random:
            correction = np.random.uniform(low, high, self.n)
        else:
            correction = np.full(self.n, per_change) if direction == "up" else np.full(self.n, -per_change)

        for r, (n, i) in zip(correction, enumerate(self.aug_img)):
            i = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
            if r > 0:
                i[:, :, 2] = np.where(i[:, :, 2].astype(float) + (r * 256) > 255, 255, i[:, :, 2] + (r * 256))
            else:
                i[:, :, 2] = np.where(i[:, :, 2].astype(float) + (r * 256) < 0, 0, i[:, :, 2] + (r * 256))
            self.aug_img[n] = cv2.cvtColor(i, cv2.COLOR_HSV2BGR)
        self.aug_steps[self.batch].append(f"brightness: per_change={per_change} direction={direction} random={random}")

    def colour(self, per_change: float = 0.1, direction: str = "both",
               bgr: tuple = ("b", "g", "r"), random: bool = True):
        assert direction in ["both", "up", "down"]
        assert 0 < len(bgr) < 4
        assert all([c in ("b", "g", "r") for c in set(bgr)])
        if per_change > 1 or per_change < 0:
            raise ValueError("Invalid percentage change")
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")

        low = -per_change if direction != "up" else 0
        high = per_change if direction != "down" else 0

        key = {"b": 0, "g": 1, "r": 2}

        for n, i in enumerate(self.aug_img):
            if random:
                correction = np.random.uniform(low, high, len(bgr))
            else:
                correction = np.full(len(bgr), per_change) if direction == "up" else np.full(len(bgr), -per_change)
            for (n, c), r in zip(enumerate(bgr), correction):
                if r > 0:
                    i[:, :, key[c]] = np.where(i[:, :, key[c]].astype(float) + (r * 256) > 255, 255,
                                               i[:, :, key[c]] + (r * 256))
                else:
                    i[:, :, key[c]] = np.where(i[:, :, key[c]].astype(float) + (r * 256) < 0, 0,
                                               i[:, :, key[c]] + (r * 256))
            self.aug_img[n] = i
        self.aug_steps[self.batch].append(f"color: per_change={per_change} direction={direction} "
                                          f"bgr={bgr} random={random}")

    def blur(self, max_ksize: int = 10, random: bool = True):
        ksize = np.round(np.random.uniform(1, max_ksize, self.n)).astype(int) if random else np.full(self.n, max_ksize)
        self.aug_img = [cv2.blur(i, (k, k)) for k, i in zip(ksize, self.aug_img)]
        self.aug_steps[self.batch].append(f"blur: max_ksize={max_ksize} random={random}")

    def sharpen(self):
        self.aug_img = [cv2.filter2D(i, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])) for i in self.aug_img]
        self.aug_steps[self.batch].append(f"sharpen")

    def flip(self, horizontal: bool = True, vertical: bool = True, random: bool = True):
        flipx = np.round(np.random.uniform(0, horizontal, self.n)) if random else np.full(self.n, horizontal)
        flipy = np.round(np.random.uniform(0, vertical, self.n)) if random else np.full(self.n, vertical)

        for x, y, b, (n, i) in zip(flipx, flipy, self.aug_bbox, enumerate(self.aug_img)):
            self.aug_img[n] = cv2.flip(i, -1) if x and y else cv2.flip(i, 1) if x else cv2.flip(i, 0) if y else i
            if x: b[:, 1] = 1 - b[:, 1]
            if y: b[:, 2] = 1 - b[:, 2]
            self.aug_bbox[n] = b
        self.aug_steps[self.batch].append(f"flip: horizontal={horizontal} vertical={vertical} random={random}")

    def rotate(self, max_rotation: float = 180, direction: str = "both", random: bool = True):
        assert direction in ["both", "clockwise", "anti-clockwise"]
        if max_rotation > 180 or max_rotation < 0:
            raise ValueError("Invalid rotation angle")
        if direction == "both" and not random:
            raise ValueError("Cannot work on both direction when not random")

        low = -max_rotation if direction != "clockwise" else 0
        high = max_rotation if direction != "anti-clockwise" else 0
        if random:
            correction = np.random.uniform(low, high, self.n)
        else:
            correction = np.full(self.n, max_rotation) if direction == "clockwise" else np.full(self.n, -max_rotation)

        for r, b, (n, i) in zip(correction, self.aug_bbox, enumerate(self.aug_img)):
            h_ratio = i.shape[0] / (np.abs(i.shape[1] * np.sin(r / 180 * np.pi)) + np.abs(i.shape[0] * np.cos(r / 180 * np.pi)))
            w_ratio = i.shape[1] / (np.abs(i.shape[1] * np.cos(r / 180 * np.pi)) + np.abs(i.shape[0] * np.sin(r / 180 * np.pi)))
            M = cv2.getRotationMatrix2D((i.shape[1] // 2, i.shape[0] // 2), r, np.min([h_ratio, w_ratio]))
            self.aug_img[n] = cv2.warpAffine(i, M, (i.shape[1], i.shape[0]))
            coords = [np.array([[(p[1] - p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] - p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1]]) for p in b]
            new_coords = [M.dot(p.T).T for p in coords]
            self.aug_bbox[n] = np.array([np.array([o[0],
                                                   (p[:, 0].max() + p[:, 0].min()) / 2 / i.shape[1],
                                                   (p[:, 1].max() + p[:, 1].min()) / 2 / i.shape[0],
                                                   (p[:, 0].max() - p[:, 0].min()) / i.shape[1],
                                                   (p[:, 1].max() - p[:, 1].min()) / i.shape[0]]) for o, p in
                                         zip(b, new_coords)])
        self.aug_steps[self.batch].append(f"rotate: max_rotation={max_rotation} direction={direction} random={random}")

    def shear(self, horizontal_max_shear: float = 0.5, vertical_max_shear: float = 0.5,
              horizontal: bool = True, vertical: bool = True,
              horizontal_direction: str = "both", vertical_direction: str = "both",
              random: bool = True):

        assert horizontal_direction in ["both", "left", "right"] and vertical_direction in ["both", "up", "down"]
        if horizontal_max_shear > 1 or horizontal_max_shear < 0 or vertical_max_shear > 1 or vertical_max_shear < 0:
            raise ValueError("Invalid rotation angle")
        if (horizontal_direction == "both" or vertical_direction == "both") and not random:
            raise ValueError("Cannot work on both direction when not random")
        if random:
            correctionx = np.random.uniform(0, horizontal_max_shear, self.n) if horizontal else np.zeros(self.n)
            correctiony = np.random.uniform(0, vertical_max_shear, self.n) if vertical else np.zeros(self.n)
            flipx = np.zeros(self.n).astype(bool) if not horizontal or horizontal_direction == "left" else\
                np.ones(self.n).astype(bool) if horizontal_direction == "right" else\
                    np.random.uniform(0, 1, self.n).round().astype(bool)
            flipy = np.zeros(self.n).astype(bool) if not vertical or vertical_direction == "up" else\
                np.ones(self.n).astype(bool) if vertical_direction == "down" else\
                    np.random.uniform(0, 1, self.n).round().astype(bool)
        else:
            correctionx = np.full(self.n, horizontal_max_shear) if horizontal else np.zeros(self.n)
            correctiony = np.full(self.n, vertical_max_shear) if vertical else np.zeros(self.n)
            flipx = np.zeros(self.n).astype(bool) if not horizontal or horizontal_direction == "left" else\
                np.ones(self.n).astype(bool)
            flipy = np.zeros(self.n).astype(bool) if not vertical or vertical_direction == "up" else\
                np.ones(self.n).astype(bool)

        for cx, cy, fx, fy, b, (n, i) in zip(correctionx, correctiony, flipx, flipy, self.aug_bbox, enumerate(self.aug_img)):
            #horizontal
            i = cv2.flip(i, 1) if fx else i
            if fx: b[:, 1] = 1 - b[:, 1]
            coords = [np.array([[(p[1] - p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] - p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1]]) for p in b]
            bounds = np.array([np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]) for p in coords])
            i = cv2.warpPerspective(i, np.array([[1, cx, 0],
                                                 [0, 1, 0],
                                                 [0, 0, 1]]),
                                    (int(i.shape[1] + i.shape[0] * cx), i.shape[0]))
            bounds[:, [0, 2]] += ((bounds[:, [1, 3]]) * cx).astype(int)
            b = np.array([np.array([int(o[0]),
                                    (p[0] + p[2]) / 2 / i.shape[1],
                                    (p[1] + p[3]) / 2 / i.shape[0],
                                    (p[2] - p[0]) / i.shape[1],
                                    (p[3] - p[1]) / i.shape[0]]) for o, p in zip(b, bounds)])
            if fx: b[:, 1] = 1 - b[:, 1]
            i = cv2.flip(i, 1) if fx else i

            #vertical
            i = cv2.flip(i, 1) if fy else i
            if fy: b[:, 2] = 1 - b[:, 2]
            coords = [np.array([[(p[1] - p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] - p[4] / 2) * i.shape[0], 1],
                                [(p[1] - p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1],
                                [(p[1] + p[3] / 2) * i.shape[1], (p[2] + p[4] / 2) * i.shape[0], 1]]) for p in b]
            bounds = np.array([np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]) for p in coords])
            i = cv2.warpPerspective(i, np.array([[1, 0, 0],
                                                 [cy, 1, 0],
                                                 [0, 0, 1]]),
                                    (i.shape[1], int(i.shape[0] + i.shape[1] * cy)))
            bounds[:, [1, 3]] += ((bounds[:, [0, 2]]) * cy).astype(int)
            b = np.array([np.array([int(o[0]),
                                    (p[0] + p[2]) / 2 / i.shape[1],
                                    (p[1] + p[3]) / 2 / i.shape[0],
                                    (p[2] - p[0]) / i.shape[1],
                                    (p[3] - p[1]) / i.shape[0]]) for o, p in zip(b, bounds)])
            if fy: b[:, 2] = 1 - b[:, 2]
            i = cv2.flip(i, 1) if fy else i
            self.aug_img[n] = i
            self.aug_bbox[n] = b
        self.aug_steps[self.batch].append(f"shear: horizontal_max_shear={horizontal_max_shear} "
                                          f"vertical_max_shear={vertical_max_shear} "
                                          f"horizontal={horizontal} vertical={vertical} "
                                          f"horizontal_direction={horizontal_direction} "
                                          f"vertical_direction={vertical_direction} "
                                          f"random={random}")

    def export(self, destination: str = ""):
        if destination != "" and not os.path.exists(destination):
            os.mkdir(destination)
        elif destination == "":
            destination = self.directory
        for name, i in zip(self.annotated, self.aug_img):
            cv2.imwrite(f"{destination}\\{name}_{self.batch}.jpg", i)
        for name, b in zip(self.annotated, self.aug_bbox):
            with open(f"{destination}\\{name}_{self.batch}.txt", "w") as coord:
                coord.write("\n".join([" ".join([str(e) if e != 0 else str(int(e)) for e in l]) for l in b.tolist()]) + "\n")
        with open(f"{destination}\\classes.txt", "w") as classes:
            classes.write(self.classes)
        try:
            with open(f"{destination}\\augmentation steps.txt", "r") as log:
                old_log = log.read()
            old_log = old_log.split("\n")[:-1]
            keys = [l.split(" -- ")[0] for l in old_log]
            items = [l.split(" -- ")[1].split("; ") for l in old_log]
            for k, i in zip(keys, items):
                if k not in self.aug_steps.keys():
                    self.aug_steps[k] = i
            self.aug_steps = {k: self.aug_steps[k] for k in sorted(self.aug_steps, key=lambda x: int(x.split("A")[-1]))}
        except FileNotFoundError:
            pass
        with open(f"{destination}\\augmentation steps.txt", "w") as log:
            log.write("\n".join([key + " -- " + "; ".join(item) for (key, item) in self.aug_steps.items()]) + "\n")

    def next(self):
        self.aug_img = copy.deepcopy(self.img)
        self.aug_bbox = copy.deepcopy(self.bbox)
        self.batch = f"A{int(self.batch[1:]) + 1}"
        self.aug_steps[self.batch] = []

    def reset(self):
        self.aug_img = copy.deepcopy(self.img)
        self.aug_bbox = copy.deepcopy(self.bbox)
        self.aug_steps[self.batch] = []
