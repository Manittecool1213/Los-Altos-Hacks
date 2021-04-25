import matplotlib.image as mpimg
import numpy as np
ls = [mpimg.imread('Screenshot (1024).png') for i in range(10)]
arr = np.array(ls)
print(arr.shape)