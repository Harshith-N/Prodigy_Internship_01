import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the training dataset
train_file_path = 'train.csv'
train_data = pd.read_csv(train_file_path)

# Check if 'SalePrice' column exists
if 'SalePrice' not in train_data.columns:
    raise KeyError("'SalePrice' column is missing from the dataset")

# Select relevant features and the target variable from the training data
features = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']].copy()  # Use .copy() to avoid chained assignment
target = train_data['SalePrice']

# Create a new feature for total bathrooms
features['TotalBath'] = features['FullBath'] + 0.5 * features['HalfBath']
features = features.drop(['FullBath', 'HalfBath'], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²) score: {r2}")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Plot Residuals vs. Predicted Prices
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted House Prices")
plt.show()

# Load the new test dataset for future prediction
test_file_path = 'test.csv'
test_data = pd.read_csv(test_file_path)

# Select the same features from the test data
test_features = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']].copy()

# Create the 'TotalBath' feature for the test data
test_features['TotalBath'] = test_features['FullBath'] + 0.5 * test_features['HalfBath']
test_features = test_features.drop(['FullBath', 'HalfBath'], axis=1)

# Predict the SalePrice using the trained model
future_predictions = model.predict(test_features)

# Add predictions to the test_data DataFrame
test_data['Predicted_SalePrice'] = future_predictions

# Save the results to a new CSV file
test_data.to_csv('/Users/harshithn/Desktop/Prodigy internship/output.csv', index=False)

# Print out the first few predictions
print(test_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'Predicted_SalePrice']].head())
