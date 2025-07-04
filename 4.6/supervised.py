# Predicting Building Energy Efficiency (Supervised Learning)

# Scenario - You are working for an architecture firm, and your task is to build a model that predicts the energy efficiency rating of buildings based on features like wall area, roof area, overall height, etc.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


warnings.filterwarnings('ignore')

# generate synthetic dataset for building features and energy efficiency rating
np.random.seed(0)
data_size = 500
data = {
    'WallArea': np.random.randint(200, 400, data_size),
    'RoofArea': np.random.randint(100, 200, data_size),
    'OverallHeight': np.random.uniform(3, 10, data_size),
    'GlazingArea': np.random.uniform(0, 1, data_size),
    'EnergyEfficiency': np.random.uniform(10, 50, data_size)
}

df = pd.DataFrame(data)
print(df)

# Data preprocessing - create 
x_axis = df.drop('EnergyEfficiency', axis = 1)
y_axis = df['EnergyEfficiency']


# visualize relationship between features and target energy efficiency
sns.pairplot(df, x_vars = ['WallArea', 'RoofArea', 'OverallHeight', 'GlazingArea'], y_vars = 'EnergyEfficiency', height = 4, aspect = 1, kind = 'scatter')

plt.show()

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size = 0.2, random_state = 42)

# train a random forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# predict and verify
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# plot the true values vs. predicted values

plt.figure(figsize = (10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()


# plot shows there's big deviation from the expected results.
# the closer the dp to the trend line, the more accurate the model is.