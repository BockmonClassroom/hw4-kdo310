import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = pd.read_csv('hw4-kdo310/Iris.csv')

#now we separate the features and the target
features = iris.drop(columns=['Species'])
species = iris['Species']

#use minmaxscaler in sklearn
normalizer = MinMaxScaler()
normalized_features = pd.DataFrame(normalizer.fit_transform(features), columns=features.columns)

#now combine the normalized features w/species
normalized_iris = pd.concat([normalized_features, species], axis=1)

#save normalized dataset
#normalized_iris.to_csv('Normalized Iris.csv', index=False)

#now to standardized
scaler = StandardScaler()
standardized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

#combining stand. features w/species
standardized_iris = pd.concat([standardized_features, species], axis=1)

#saving the dataset
standardized_iris.to_csv('Standardized Iris.csv', index=False)