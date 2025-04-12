from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle

def train_RFmodel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=123)
    model.fit(X_train_scaled, y_train)
    with open('models/RFmodel.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model, scaler, X_test_scaled, y_test
