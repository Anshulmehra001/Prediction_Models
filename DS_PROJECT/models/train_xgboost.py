import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'objective': 'reg:squarederror'
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model