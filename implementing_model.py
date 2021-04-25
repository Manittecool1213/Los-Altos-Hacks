import pickle
import numpy as np
from numpy.core.defchararray import asarray
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
from PIL import ImageEnhance
import cv2


loaded_model = pickle.load(open('RandomForestModel.sav', 'rb'))


img = np.array([i - 30 for i in cv2.imread('output.png', 0)]).reshape(-1)
print(img.shape)
# img.thumbnail((28, 28), Image.ANTIALIAS)  # resizes image in-place

# enhancer = ImageEnhance.Contrast(img)
# image = enhancer.enhance(1.0)

# im = np.array(img.convert('L').resize((28, 28))) #you can pass multiple arguments in single line
# print(type(im))

#############################################################


# data = np.asarray(im).reshape(-1)

# data = data.flatten()
# print(data.shape)
# data = data.transpose(2,0,1).reshape(3,-1)
# data = np.array([data])

# data = np.array([data, data])

class ImageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, width=14, height=14,  is_img=False):
        self.width = width
        self.height = height
        self.is_img = is_img

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.is_img:
            images = []
            # load the image and convert to grayscale
            for img in X:

                image = img.convert('L').resize((self.width, self.height))

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.0)

                images.append(np.asarray(image).reshape(-1))
            return np.array(images)
        else:
            return X

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

preprocessing_pipeline = Pipeline([
    ('image_trf', ImageTransformer()),
    ('scaler', StandardScaler()),
])

width, height = 28, 28
preprocessing_pipeline.set_params(image_trf__width=width, image_trf__height=height, image_trf__is_img=False)

ls = []
ls.append(img)
ls.append(img)
ls = np.array(ls)

# data = preprocessing_pipeline.fit_transform(ls)

result = loaded_model.predict(ls)

english_alphabet = pd.DataFrame(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                 'u', 'v', 'w', 'x', 'y', 'z'], [i for i in range(1, 27)])


# print(english_alphabet.loc[result])
print(result)







