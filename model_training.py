import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def train_model(input_csv, model_output):

    df = pd.read_csv(input_csv)

    # --------------------------------
    # create prediction target
    # next month spending
    # --------------------------------
    df["target"] = df.groupby("category")["spending"].shift(-1)

    # remove rows without target
    df = df.dropna()

    # --------------------------------
    # feature columns
    # --------------------------------
    features = [
        "lag1",
        "lag2",
        "lag3",
        "rolling_avg",
        "rolling_std",
        "trend",
        "momentum",
        "month_sin",
        "month_cos"
    ]

    X = df[features]
    y = df["target"]

    # --------------------------------
    # train test split
    # --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------
    # train random forest
    # --------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------
    # predictions
    # --------------------------------
    preds = model.predict(X_test)

    # --------------------------------
    # evaluation
    # --------------------------------
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Model Evaluation")
    print("MAE:", mae)
    print("R2 Score:", r2)

    # --------------------------------
    # save model
    # --------------------------------
    joblib.dump(model, model_output)

    print("Model saved as:", model_output)

    return model