import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

TRAIN_DATA_FILE_PATH = './train_mapped.csv'

all_train_data = pd.read_csv(TRAIN_DATA_FILE_PATH)

# Splitting input and output
data_train = [all_train_data["Expected"]]
headers_train = ["Expected"]
split_train = pd.concat(data_train, axis=1, keys=headers_train)

all_train_data.drop(['Expected'], axis = 1, inplace=True) 
X = pd.get_dummies(all_train_data)
y = split_train["Expected"]

print("Feature selection process may take long time...")

# Setting up RFE
rfe = RFE(estimator=DecisionTreeClassifier(random_state=1), n_features_to_select=100, step=0.65)
rfe = rfe.fit(X, y)

# Retrieve list of features from the index of features selected by RFE.
list_x = []
col_names = []
selected_features = []

list_x = rfe.get_support(indices=True)
col_names = X.columns.tolist()

for i in list_x: 
    selected_features.append(col_names[i])	
	
print("Selected features:")
print(selected_features)
print("Done...")
