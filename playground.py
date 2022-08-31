import os, glob

self.aug_img = [cv2.filter2D(i, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])) for i in self.aug_img]
self.aug_steps[self.batch].append(f"sharpen")
self.export("F:\\mphil\\Cheryl's folder\\testing4")
self.next()

k_size = 5
k = np.full((k_size, k_size), -1)
k[k_size // 2 - 1, k_size // 2 - 1] = k_size ** 2
self.aug_img = [cv2.filter2D(i, -1, k) for i in self.aug_img]
self.aug_steps[self.batch].append(f"sharpen")
self.export("F:\\mphil\\Cheryl's folder\\testing5")
self.next()