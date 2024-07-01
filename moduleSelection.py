import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = "~/data-sets/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
data_cols = melbourne_data.columns
print(data_cols)

#  axis = 0 refers to horizontal axis or rows and axis = 1 refers to vertical axis or columns
melbourne_data = melbourne_data.dropna(axis=0)
print(type(melbourne_data))

# what to predict
y = melbourne_data.Price
print(y.head())

# inputs to perdict
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())

"""
The steps to building and using a model are:
DFPE:

Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
Fit: Capture patterns from provided data. This is the heart of modeling.
Predict: Just what it sounds like
Evaluate: Determine how accurate the model's predictions are.
"""

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for.
# But we'll make predictions for the first few rows of the training data to see how the predict function works.
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
