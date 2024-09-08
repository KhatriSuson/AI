import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
#Load Dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target
# Explore Data
print (data.head())
print(data.describe())

# Visualize the Data
sns.pairplot(data, x_vars=['RM', 'LSTAT', 'PTRATION'], y_vars='PRICE', height=7, aspect=0.7, kind='reg')
plt.show()

# Prepare the data
x = data.drop('PRICE', axis=1)
y = data['PRICE']
x_train, x_test, y_train, t_test = train_test_split(x, y, test_size=0.2, random_state=43)

# Train the Model
model = LinearRegression()
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error :{mse:.2f}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()


print(pd.__version__)
