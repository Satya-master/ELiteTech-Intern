import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate a synthetic dataset
# For demonstration, let's create a dataset with 100 samples
np.random.seed(42)  # For reproducibility

# Generate random square footage (500 to 3500 sq ft)
square_footage = np.random.randint(500, 3500, 100)

# Generate random number of bedrooms (1 to 5)
bedrooms = np.random.randint(1, 6, 100)

# Generate random number of bathrooms (1 to 3)
bathrooms = np.random.randint(1, 4, 100)

# Generate a random price based on a simple linear relationship with some noise
price = (square_footage * 300) + (bedrooms * 5000) + (bathrooms * 3000) + np.random.randint(-20000, 20000, 100)

# Create a DataFrame
data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': price
})

# Step 2: Split the dataset into training and testing sets
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions using the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print the coefficients of the model
print("Model Coefficients:")
print(f"Square Footage Coefficient: {model.coef_[0]}")
print(f"Bedrooms Coefficient: {model.coef_[1]}")
print(f"Bathrooms Coefficient: {model.coef_[2]}")
print(f"Intercept: {model.intercept_}")

# Example prediction
example_house = np.array([[2000, 3, 2]])  # 2000 sq ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(example_house)
print(f"Predicted Price for the example house: ${predicted_price[0]:.2f}")
