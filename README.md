# Machine-learning

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example dataset: features and target
data = {
    'sqft': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 3, 2, 4, 3],
    'price': [400000, 500000, 600000, 650000, 700000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define features and target variable
X = df[['sqft', 'bedrooms', 'bathrooms']]  # Features
y = df['price']                             # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print coefficients
print("Model coefficients:")
print(f"  sqft coefficient: {model.coef_[0]}")
print(f"  bedrooms coefficient: {model.coef_[1]}")
print(f"  bathrooms coefficient: {model.coef_[2]}")
print(f"Intercept: {model.intercept_}")

# Example prediction (using DataFrame to avoid warning)
new_house = pd.DataFrame([[2000, 3, 2]], columns=['sqft', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(new_house)
print("\n")

print(f"Predicted price for house with 2000 sqft, 3 bedrooms, 2 bathrooms: ${predicted_price[0]:,.2f}")
print("\n")
