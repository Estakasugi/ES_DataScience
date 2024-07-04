import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

melbourne_file_path = "~/data-sets/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]


# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)

# this is a in-sample score
"""
Since models' practical value come from making predictions on new data, 
we measure performance on data that wasn't used to build the model.

The most straightforward way to do this is to 
exclude some data from the model-building process, and then use those to test 
the model's accuracy on data it hasn't seen before. 
This data is called validation data
"""
perdicted_home_price = melbourne_model.predict(X)
mae_exercise = mean_absolute_error(y, perdicted_home_price)
print(mae_exercise)

