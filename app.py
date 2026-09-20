import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = r"C:\Users\IT SHOP\OneDrive\Documents\Default Project\car_price_model.pkl"
FEATURES_PATH = r"C:\Users\IT SHOP\OneDrive\Documents\Default Project\model_features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

st.set_page_config(page_title="Car Price Predictor", page_icon="car")
st.title("Car Price Predictor")
st.caption("Predict the selling price of a used car in Lakhs.")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input(
        "Present Price (Lakhs)", min_value=0.0, value=8.5, step=0.1)
    year = st.number_input("Year of purchase", min_value=2000,
                           max_value=2026, value=2020, step=1)
    kms = st.number_input("Kms driven", min_value=0, value=40000, step=1000)
    owners = st.selectbox("Number of owners", [0, 1, 2, 3, 4])

with col2:
    fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
    seller = st.selectbox("Seller type", ["Dealer", "Individual"])
    gearbox = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price", type="primary"):
    row = {
        "Present_Price": present_price,
        "Car_Age": 2026 - int(year),
        "Kms_Driven": kms,
        "Owner": owners,
        "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
        "Fuel_Type_Petrol": 1 if fuel == "Petrol" else 0,
        "Seller_Type_Individual": 1 if seller == "Individual" else 0,
        "Transmission_Manual": 1 if gearbox == "Manual" else 0,
    }
    input_df = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
    price = float(model.predict(input_df)[0])
    st.success(f"Predicted Selling Price: {price:,.2f} Lakhs")