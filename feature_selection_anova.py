'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca
This script does Feature ranking with ANOVA.

Output:
List of features after ANOVA strategy

For tuning RFE refer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

TRAIN_DATA_FILE_PATH = './train_mapped.csv'

# Reading Train and Test Data files
all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)

# Encode chemical IDS
train_id =  list(all_train_data.Id.unique())

le = preprocessing.LabelEncoder()
le.fit(train_id)
all_train_data['Id']=le.transform(all_train_data['Id'])

# Splitting input and output
data_train = [all_train_data["Expected"]]
headers_train = ["Expected"]
split_train = pd.concat(data_train, axis=1, keys=headers_train)
all_train_data.drop(labels=["Expected"], axis = 1, inplace=True)

# Get test and train data on the basis of the feature. Also, and get target data.
X = pd.get_dummies(all_train_data)
y = split_train["Expected"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# select the number of features you want to retain.
select_k =50
numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]

# create the SelectKBest with the Anova strategy.
selection = SelectKBest(f_classif, k=select_k).fit(X_train, y_train)

# display the retained features.
features = X_train.columns[selection.get_support()]

print(features)

print("Done...")
