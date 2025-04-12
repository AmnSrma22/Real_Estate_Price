# 🏠 Real Estate Price Forecast App
View the app from here "https://realestateprice-elo3njhzhwvu5sngyvggfb.streamlit.app/"

This application provides housing price estimates based on property characteristics using a machine learning model deployed through a user-friendly Streamlit interface.

## 📌 Highlights
- Estimates home value using Random Forest regression
- Accepts details like size, location type, construction year, and more
- Generates feature importance charts
- Displays error metrics (RMSE, R²)
- Trained model saved for consistent reuse

## 🔍 Contents
- `streamlit.py`: Web interface
- `main.py`: Script to train and evaluate the model
- `models/`: Folder where the trained model is saved
- `src/`: Code modules for cleaning, transforming, modeling, and visualizing data

## ▶️ Launch the App
```bash
streamlit run streamlit.py
```

## ⚙️ Dependencies
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn