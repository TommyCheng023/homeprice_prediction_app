import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np

model = joblib.load("model.pkl")
required_features = model.named_steps["preprocessor"].feature_names_in_

def geocode_address(address):
    api_key = st.secrets["POSITIONSTACK_API_KEY"]
    base_url = "http://api.positionstack.com/v1/forward"
    params = {"access_key": api_key, "query": address, "limit": 1}
    try:
        resp = requests.get(base_url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            loc = data["data"][0]
            return loc["latitude"], loc["longitude"]
        else:
            return None, None
    except:
        return None, None

# App layout
st.set_page_config(page_title="🏡 House Price Predictor", layout="wide")
st.title("🏡 California House Price Predictor")

# Mode selector
mode = st.radio("Select Input Mode", ["🟢 Simple Mode", "⚙️ Full Mode"])

# Shared field
address = st.text_input("Address", "1600 Amphitheatre Parkway, Mountain View, CA")

if mode == "🟢 Simple Mode":
    st.subheader("🟢 Basic Property Info")

    living_area = st.number_input("Living Area (sq ft)", min_value=200, max_value=10000, value=1500)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    stories = st.number_input("Stories", min_value=1, max_value=5, value=1)
    garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=5, value=2)
    lot_size_sqft = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=5000)

    view_yn = st.checkbox("Has View")
    pool_yn = st.checkbox("Private Pool")
    attached_garage_yn = st.checkbox("Attached Garage")
    fireplace_yn = st.checkbox("Fireplace")
    new_construction_yn = st.checkbox("New Construction")

elif mode == "⚙️ Full Mode":
    st.subheader("⚙️ Full Feature Info")
    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("BedroomsTotal", min_value=0, value=3)
        bathrooms = st.number_input("BathroomsTotalInteger", min_value=0, value=2)
        stories = st.number_input("Stories", min_value=1, value=1)
        fireplace_total = st.number_input("FireplacesTotal", value=1)
        garage_spaces = st.number_input("GarageSpaces", value=2)
        parking_total = st.number_input("ParkingTotal", value=2)
        lot_size_sqft = st.number_input("LotSizeSquareFeet", value=5000)
        living_area = st.number_input("LivingArea", value=1500)

    with col2:
        age = st.number_input("Age", value=20)
        lot_density = st.number_input("LotDensity", value=lot_size_sqft / 43560.0)
        sales_tax = st.number_input("SalesTaxRate", value=0.075)
        unrate = st.number_input("UNRATE", value=0.037)
        mortgage_rate = st.number_input("MORTGAGE30US", value=0.066)
        fed_funds = st.number_input("FEDFUNDS", value=0.053)
        cpi = st.number_input("CPIAUCNS", value=305.1)

    with col3:
        view_yn = st.checkbox("ViewYN")
        pool_yn = st.checkbox("Private Pool")
        attached_garage_yn = st.checkbox("Attached Garage")
        fireplace_yn = st.checkbox("Fireplace")
        new_construction_yn = st.checkbox("New Construction")

        city = st.text_input("City", "Mountain View")
        postal = st.text_input("PostalCode", "94043")
        county = st.text_input("CountyOrParish", "Santa Clara")
        state = st.text_input("StateOrProvince", "CA")
        property_subtype = st.text_input("PropertySubType", "SingleFamilyResidence")
        levels = st.selectbox("Levels", ["One", "Two", "Three", "Split", "Other"], index=0)
        district = st.text_input("District", "Mountain View")
        highschool = st.text_input("HighSchoolDistrict", "Mountain View-Los Altos Union")
        flooring = st.selectbox("FlooringType", ["Hardwood", "Tile", "Carpet", "Laminate", "Vinyl"], index=0)

# Predict Button
if st.button("Predict Price"):
    with st.spinner("Geocoding and predicting..."):
        lat, lon = geocode_address(address)
        if lat is None or lon is None:
            st.error("Geocoding failed.")
            st.stop()

        # Build input_data by mode
        if mode == "🟢 Simple Mode":
            input_data = {
                "Latitude": [lat],
                "Longitude": [lon],
                "LivingArea": [living_area],
                "BathroomsTotalInteger": [bathrooms],
                "BedroomsTotal": [bedrooms],
                "Stories": [stories],
                "GarageSpaces": [garage_spaces],
                "LotSizeSquareFeet": [lot_size_sqft],
                "ViewYN": [int(view_yn)],
                "PoolPrivateYN": [int(pool_yn)],
                "AttachedGarageYN": [int(attached_garage_yn)],
                "FireplaceYN": [int(fireplace_yn)],
                "NewConstructionYN": [int(new_construction_yn)],
                "SalesTaxRate": [0.075],
                "UNRATE": [0.037],
                "MORTGAGE30US": [0.066],
                "FEDFUNDS": [0.053],
                "CPIAUCNS": [305.1],
                "Age": [20],
                "LotDensity": [lot_size_sqft / 43560.0],
                "ParkingTotal": [garage_spaces],
                "FireplacesTotal": [1],
                "City": ["Mountain View"],
                "CountyOrParish": ["Santa Clara"],
                "PostalCode": ["94043"],
                "Levels": ["One"],
                "StateOrProvince": ["CA"],
                "PropertySubType": ["SingleFamilyResidence"],
                "HighSchoolDistrict": ["Mountain View-Los Altos Union"],
                "District": ["Mountain View"],
                "FlooringType": ["Hardwood"]
            }
        else:
            input_data = {
                "Latitude": [lat],
                "Longitude": [lon],
                "BedroomsTotal": [bedrooms],
                "BathroomsTotalInteger": [bathrooms],
                "Stories": [stories],
                "FireplacesTotal": [fireplace_total],
                "GarageSpaces": [garage_spaces],
                "ParkingTotal": [parking_total],
                "LotSizeSquareFeet": [lot_size_sqft],
                "LivingArea": [living_area],
                "Age": [age],
                "LotDensity": [lot_density],
                "SalesTaxRate": [sales_tax],
                "UNRATE": [unrate],
                "MORTGAGE30US": [mortgage_rate],
                "FEDFUNDS": [fed_funds],
                "CPIAUCNS": [cpi],
                "ViewYN": [int(view_yn)],
                "PoolPrivateYN": [int(pool_yn)],
                "AttachedGarageYN": [int(attached_garage_yn)],
                "FireplaceYN": [int(fireplace_yn)],
                "NewConstructionYN": [int(new_construction_yn)],
                "City": [city],
                "CountyOrParish": [county],
                "PostalCode": [postal],
                "Levels": [levels],
                "StateOrProvince": [state],
                "PropertySubType": [property_subtype],
                "HighSchoolDistrict": [highschool],
                "District": [district],
                "FlooringType": [flooring]
            }

        df = pd.DataFrame(input_data)

        # Ensure proper types
        string_cols = ['City', 'CountyOrParish', 'PostalCode', 'Levels', 'StateOrProvince', 'PropertySubType', 'HighSchoolDistrict', 'District', 'FlooringType']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df[required_features]

        # Predict and inverse log
        log_price = model.predict(df)[0]
        price = np.exp(log_price)

        st.success(f"🏷️ Estimated Close Price: **${price:,.0f}**")
        if mode == "🟢 Simple Mode":
            st.info("💡 Process done under simple mode. Consider switching to Full Mode for more accurate predictions.")