"""
Import Dependencies:
1. Matplotlib
2. Numpy
3. Pandas
4. Sklearn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from PIL import ImageEnhance, Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""
Exploring Data
1. Opening the data as a panda data frame, then converting to a numpy array
2. Dividing the data into dependant and independant variables
3. Dividing the data into training and testing data

english_dataset
Type: numpy array
Description: stores over 37,000 data elements of writing examples along with their true values

X
Type: numpy nd-array
Description: dependant variables, used to determine target values (y)

y
Type: numpy nd-array
Description: target values

X_train, y_train
Type: numpy nd-arrays
Description: trainging values

X_test, y_test
Type: numpy nd-arrays
Description: testing values
"""
english_dataset = np.array(pd.read_csv('C:/Users/tanwa/Downloads/A_Z Handwritten Data/A_Z Handwritten Data.csv')) # The path can be replaced with the local path containing the data file.

X, y = english_dataset[:, 1:], english_dataset[:, 0] # Dividing the data into dependant and independant variables.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1) # Dividing the data into training and testing data.


new_arr = X_test[0]
new_arr.shape = (28, 28)

imgplot = plt.imshow(new_arr)
plt.show()

im = Image.fromarray(X_test[0])
im.save('from-array-p.png')

print(y_test[0])

"""
english_alphabet = pd.DataFrame(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                'u', 'v', 'w', 'x', 'y', 'z'], [i for i in range(1, 27)])


def plot_letters(images, labels, width=14, height=14):
    rows, cols = 4, 6

    fig=plt.figure(figsize=(14, 14))
    sub_plot_i = 1

    for i in range(0, 2):
        fig.add_subplot(rows, cols, sub_plot_i)
        sub_plot_i += 1
        image = images[i].reshape(width, height)
        plt.imshow(image, cmap='gray')
        label = labels[i].astype(int) + 1
        plt.title(english_alphabet.loc[label][0])


    fig.tight_layout()
    plt.show()

plot_letters(X_train, y_train, 28, 28)
"""


# Preparing data:
# 1. Image transformer:
#     a) Resizes the image to width(int) and height(int)
#     b) Enhances contrast in the image(converts to greyscale)
# 2. Preprocessing pipeline:
#     Recieves instance of ImageTransformer and StandardScaler(builtin sklearn class) and applies both to x_train to form
#     x_train_proc which is the training data used for the model.
"""
from sklearn.base import BaseEstimator, TransformerMixin

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
X_train_proc = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

"""
# Implementing random forestclassifier
# Accuracy ~ 90%
"""
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(max_depth=10)
forest_clf.fit(X_train_proc[:25000], y_train[:25000])

# from sklearn.metrics import accuracy_score
# y_pred_forest = forest_clf.predict(X_test)
# print(accuracy_score(y_pred_forest, y_test))

# # Saving model using pickle
# filename = 'RandomForestModel.sav'
# pickle.dump(forest_clf, open(filename, 'wb'))

# img =  Image.open('Screenshot (1024).png')
# img.thumbnail((28, 28), Image.ANTIALIAS)  # resizes image in-place

# enhancer = ImageEnhance.Contrast(img)
# image = enhancer.enhance(1.0)

# im = np.array(img.convert('L').resize((28, 28))) #you can pass multiple arguments in single line
# print(type(im))

#############################################################
# ls = [mpimg.imread('Screenshot (1024).png') for i in range(10)]


# data = np.asarray(im).reshape(-1)

# data = data.flatten()
# print(data.shape)
# data = data.transpose(2,0,1).reshape(3,-1)
# data = np.array([data])

# data = np.array([data, data])

# data = preprocessing_pipeline.fit_transform(np.array(ls))

# result = forest_clf.predict(data)

# english_alphabet = pd.DataFrame(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
#                                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
#                                 'u', 'v', 'w', 'x', 'y', 'z'], [i for i in range(1, 27)])


# # print(english_alphabet.loc[result])
# print(result)

"""

