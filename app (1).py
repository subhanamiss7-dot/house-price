import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# -----------------------------
# 1. Create Sample Dataset
# -----------------------------
data = {
    "Size": [500, 800, 1000, 1200, 1500, 1800, 2000],
    "Price": [50000, 80000, 100000, 120000, 150000, 180000, 200000]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Prepare Features & Target
# -----------------------------
X = df[["Size"]]   # Independent variable
y = df["Price"]    # Dependent variable

# -----------------------------
# 3. Train Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 4. Save Model using joblib
# -----------------------------
model_file = "house_price_model.pkl"
joblib.dump(model, model_file)

# -----------------------------
# 5. Streamlit App UI
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("This app predicts house prices based on house size (sq ft).")

# User input
house_size = st.number_input(
    "Enter House Size (in sq ft):",
    min_value=100,
    max_value=10000,
    value=1000,
    step=50
)

# Predict button
if st.button("Predict Price"):
    size_array = np.array([[house_size]])
    loaded_model = joblib.load(model_file)
    prediction = loaded_model.predict(size_array)
    st.success(f"💰 Predicted House Price: {prediction[0]:,.2f}")
