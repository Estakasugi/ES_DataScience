import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# load data
melbourne_file_path = "~/data-sets/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

# filter data
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# set target
y = filtered_melbourne_data.Price

# set feature
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]


# train validation data split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# define a model
melbourne_model = DecisionTreeRegressor()

# fit the model
melbourne_model.fit(train_X, train_y)

# perdict
val_perdiction = melbourne_model.predict(val_X) # the input argument of perdict method is X(selected feature datasets)
print(mean_absolute_error(val_y, val_perdiction))
