import streamlit as st
import pandas as pd
import pickle
from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
from src.models.visualization import plot_feature_importance
from src.features.build_features import build_features
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
        }
        .sub {
            color: #555;
            font-size: 18px;
            margin-top: -10px;
        }
        .card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .result {
            font-size: 28px;
            font-weight: 600;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏠 Real Estate Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Enter property details to estimate its market value</p>", unsafe_allow_html=True)

model, scaler = None, None
with st.sidebar:
    st.header("📁 Load & Train Model")
    if st.button("⚙️ Train Model"):
        with st.spinner("Training model..."):
            df = load_and_preprocess_data("final.csv")
            X = build_features(df.drop("price", axis=1))
            y = df["price"]
            model, scaler, X_test_scaled, y_test = train_RFmodel(X, y)
            rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
            st.success("✅ Training Complete!")
            st.metric("RMSE", f"${rmse:,.2f}")
            st.metric("R² Score", f"{r2:.2f}")
            st.markdown("### 🔍 Feature Importance")
            plot_feature_importance(model, X)


try:
    with open("models/RFmodel.pkl", "rb") as f:
        model = pickle.load(f)
    df = load_and_preprocess_data("final.csv")
    X = build_features(df.drop("price", axis=1))
    scaler = MinMaxScaler().fit(X)
except FileNotFoundError:
    st.warning("⚠️ Please train the model first using the sidebar.")

# ---------- Input Form ----------
st.markdown("### 🏡 Enter Property Details")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        year_sold = st.number_input("📆 Year Sold", min_value=2000, max_value=2025, value=2023)
        property_tax = st.number_input("💸 Property Tax ($)", min_value=0.0, value=3000.0)
        insurance = st.number_input("🛡️ Insurance Cost ($)", min_value=0.0, value=1200.0)

    with col2:
        beds = st.number_input("🛏️ Bedrooms", min_value=0, value=3)
        baths = st.number_input("🛁 Bathrooms", min_value=0, value=2)
        sqft = st.number_input("📐 Square Feet", min_value=100, value=1500)

    with col3:
        year_built = st.number_input("🏗️ Year Built", min_value=1800, max_value=2025, value=2005)
        lot_size = st.number_input("🌳 Lot Size (sqft)", min_value=0.0, value=5000.0)
        basement = st.selectbox("🏠 Basement", ["None", "Finished", "Unfinished"])
        property_type = st.selectbox("🏘️ Property Type", ["House", "Condo", "Townhouse"])

    submit = st.form_submit_button("🎯 Predict Price")

if submit and model and scaler is not None:
    input_df = pd.DataFrame([{
        'year_sold': year_sold,
        'property_tax': property_tax,
        'insurance': insurance,
        'beds': beds,
        'baths': baths,
        'sqft': sqft,
        'year_built': year_built,
        'lot_size': lot_size,
        'basement': basement,
        'property_type': property_type
    }])

    # Append to training set to preserve encoding pipeline
    full_df = load_and_preprocess_data("final.csv")
    full_df = pd.concat([full_df.drop(columns=["price"], errors='ignore'), input_df], ignore_index=True)
    encoded = build_features(full_df)
    user_encoded = encoded.tail(1)
    user_scaled = scaler.transform(user_encoded)

    prediction = model.predict(user_scaled)[0]

    st.markdown("---")
    st.subheader("💰 Predicted Price")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='card'><div class='result'>Estimated Property Value:</div><br><h2>${prediction:,.2f}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.progress(min(int((prediction/1_000_000)*100), 100), text="Relative Price Scale")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Random Forests")
