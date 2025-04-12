import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
from src.features.build_features import build_features

if __name__ == "__main__":
    df = load_and_preprocess_data("final.csv")
    X = build_features(df.drop("price", axis=1))
    y = df["price"]
    model, scaler, X_test_scaled, y_test = train_RFmodel(X, y)
    rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
    print("Model RMSE:", rmse)
    print("R2 Score:", r2)
