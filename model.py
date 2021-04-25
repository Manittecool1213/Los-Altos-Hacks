"""
Import Dependencies:
1. Matplotlib
2. Numpy
3. Pandas
4. Sklearn
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


"""
Exploring Data
1. Opening the data as a panda data frame, then converting to a numpy array
2. Dividing the data into dependant and independant variables
3. Dividing the data into training and testing data

english_dataset
Type: numpy array
Description: stores over 37,000 data elements of writing examples along with their true values



"""
english_dataset = np.array(pd.read_csv('C:/Users/tanwa/Downloads/A_Z Handwritten Data/A_Z Handwritten Data.csv')) # The path can be replaced with the local path containing the data file.

X, y = english_dataset[:, 1:], english_dataset[:, 0] # Dividing the data into dependant and independant variables.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Dividing the data into training and testing data.

# Preparing Data
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


# Implementing Random Forest
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(max_depth=10)
forest_clf.fit(X_train_proc[:25000], y_train[:25000])

# from sklearn.metrics import accuracy_score
# y_pred_forest = forest_clf.predict(X_test)
# print(accuracy_score(y_pred_forest, y_test))

# Saving model using pickle
filename = 'RandomForestModel.sav'
pickle.dump(forest_clf, open(filename, 'wb'))
