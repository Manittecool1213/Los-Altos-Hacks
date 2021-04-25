import pickle
import numpy as np
from numpy.core.defchararray import asarray
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
from PIL import ImageEnhance


loaded_model = pickle.load(open('RandomForestModel.sav', 'rb'))


img = Image.open('Screenshot (1024).png')
# img.thumbnail((28, 28), Image.ANTIALIAS)  # resizes image in-place

# enhancer = ImageEnhance.Contrast(img)
# image = enhancer.enhance(1.0)

im = np.array(img.convert('L').resize((28, 28))) #you can pass multiple arguments in single line
# print(type(im))

#############################################################


data = np.asarray(im).reshape(-1)

# data = data.flatten()
# print(data.shape)
# data = data.transpose(2,0,1).reshape(3,-1)
# data = np.array([data])

data = np.array([data, data])

result = loaded_model.predict(data)

english_alphabet = pd.DataFrame(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                'u', 'v', 'w', 'x', 'y', 'z'], [i for i in range(1, 27)])


# print(english_alphabet.loc[result])
print(result)







