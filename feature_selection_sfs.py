'''
Author: Team NextGen x2020flg@stfx.ca, x2020fli@stfx.ca, x2020fle@stfx.ca
This script does Feature ranking with Sequential Feature Selector.

Output:
List of features after Sequential Feature Selector

For tuning SFS refer: http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


TRAIN_DATA_FILE_PATH = './train_mapped.csv'

all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)

# Convert string values of ID using lable encoder
le = preprocessing.LabelEncoder()
train_id =  list(all_train_data.Id.unique())
le.fit(train_id)
all_train_data['Id']=le.transform(all_train_data['Id'])

# Splitting input and output
data_train = [all_train_data["Expected"]]
headers_train = ["Expected"]
split_train = pd.concat(data_train, axis=1, keys=headers_train)

all_train_data.drop(labels=["Expected"], axis = 1, inplace=True)
X = pd.get_dummies(all_train_data)
y = split_train["Expected"]

print("Sequential Feature Selection process may take long time...")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

# Setting up SFS for DecisionTree
clf = DecisionTreeClassifier()

sfs1 = sfs(clf,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='f1_macro',
           cv=5)

sfs1.fit(X_train, y_train)

print('Best accuracy score: %.2f' % sfs1.k_score_)   # k_score_ shows the best score
print('Best subset (indices):', sfs1.k_feature_idx_) # k_feature_idx_ shows the index of features
print('Best subset (corresponding names):', sfs1.k_feature_names_) # k_feature_names_ shows the feature names

feature_cols = pd.DataFrame(sfs1.subsets_).transpose()
print(feature_cols)
