import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load Dataset

data = pd.read_csv('kc_house_prediction')

# Data Preprocessing
data.ffill(inplace=True)  # Handle missing values

# Drop unnecessary columns
data.drop(columns=['date'], inplace=True)

# Define features and target
X = data[['sqft_living', 'bedrooms', 'bathrooms']]  # Correctly select features
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definitions
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}


# Train and Evaluate Models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {
        "Mean Squared Error": mse,
        "R-squared": r2
    }

# Streamlit App
st.title("House Price Prediction")
st.sidebar.title("Input Features")

# Sidebar for User Input
area = st.sidebar.number_input("Enter Area (sqft):", value=2000, step=100)
bedrooms = st.sidebar.number_input("Enter Number of Bedrooms:", value=3, step=1)
bathrooms = st.sidebar.number_input("Enter Number of Bathrooms:", value=2, step=1)


# Predict Button
if st.sidebar.button("Predict"):
    user_input = pd.DataFrame({
        'sqft_living': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms] ,
     
    })
    

    # Ensure user_input aligns with the trained model's features
    for model_name, model in models.items():
        prediction = model.predict(user_input)
        st.write(f"Predicted Price using {model_name}: ${prediction[0]:,.2f}")

# Display Results of Models
st.write("### Model Evaluation")
for model_name, metrics in results.items():
    st.write(f"#### {model_name}")
    st.write(f"- Mean Squared Error: {metrics['Mean Squared Error']}")
    st.write(f"- R-squared: {metrics['R-squared']}")

