from pandas import read_csv
from numpy import NaN
from sklearn.model_selection import train_test_split

drop_col = 'BI-RAIDS'
target_col = 'Severity'
features = ['Age', 'Shape', 'Margin', 'Density']
names = [drop_col] + features + [target_col]
# read the mammographic masses dataset
dataframe = read_csv('mammographic_masses.data.txt', names=names)
# we replace the values marked as '?' which are unknown as numpy NaN values
dataframe = dataframe.replace('?', NaN)
# we drop the first column which is irrelevant
dataframe.drop(labels=[drop_col], axis=1)
# let's drop the Nan values
dataframe = dataframe.dropna()

X = dataframe[features]
y = dataframe[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
