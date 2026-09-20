import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\IT SHOP\OneDrive\Documents\Default Project\car data.csv"
MODEL_PATH = r"C:\Users\IT SHOP\OneDrive\Documents\Default Project\car_price_model.pkl"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)
print(f"rows after drop_duplicates: {len(df)}")

CAT_COLS = ["Fuel_Type", "Seller_Type", "Transmission"]


def baseline():
    print("\n========== PART A: REPRODUCING YOUR NOTEBOOK (baseline) ==========")
    X = df.drop(columns=["Car_Name", "Selling_Price"])
    y = df["Selling_Price"]
    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    results = []
    for name, m in models.items():
        if name == "LinearRegression":
            pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
        else:
            pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "model": name, "MAE": round(mean_absolute_error(y_test, pred), 3),
            "RMSE": round(mean_squared_error(y_test, pred) ** 0.5, 3),
            "R2": round(r2_score(y_test, pred), 3),
        })
    return pd.DataFrame(results)


def improved():
    print("\n========== PART B: IMPROVED PIPELINE ==========")
    work = df.copy()
    work["Car_Age"] = 2026 - work["Year"]
    work = work.drop(columns=["Car_Name", "Year"])
    X = work.drop(columns=["Selling_Price"])
    y = work["Selling_Price"]
    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    rf_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["auto", "sqrt"],
    }
    gb_space = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
        "subsample": [0.8, 1.0],
    }
    et_space = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }

    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(random_state=42)
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)

    tuned = {
        "DecisionTree(tuned)": GridSearchCV(
            dt, {"max_depth": [5, 8, 12, None], "min_samples_leaf": [1, 2, 5]},
            cv=5, scoring="r2"),
        "RandomForest(tuned)": RandomizedSearchCV(
            rf, rf_space, n_iter=20, cv=5, scoring="r2", random_state=42, n_jobs=-1),
        "GradientBoosting(tuned)": RandomizedSearchCV(
            gb, gb_space, n_iter=20, cv=5, scoring="r2", random_state=42, n_jobs=-1),
        "ExtraTrees(tuned)": RandomizedSearchCV(
            et, et_space, n_iter=15, cv=5, scoring="r2", random_state=42, n_jobs=-1),
    }

    plain = {
        "LinearRegression": LinearRegression(),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    results = []
    for name, m in tuned.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        cv = cross_val_score(m.best_estimator_, X_train, y_train, cv=5,
                             scoring="r2")
        results.append({
            "model": name, "CV_R2(mean)": round(cv.mean(), 3),
            "best_params": str(m.best_params_),
            "MAE": round(mean_absolute_error(y_test, pred), 3),
            "RMSE": round(mean_squared_error(y_test, pred) ** 0.5, 3),
            "R2": round(r2_score(y_test, pred), 3),
        })

    for name, m in plain.items():
        if name == "LinearRegression":
            pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
        else:
            pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        results.append({
            "model": name, "CV_R2(mean)": round(cv.mean(), 3),
            "best_params": "-", "MAE": round(mean_absolute_error(y_test, pred), 3),
            "RMSE": round(mean_squared_error(y_test, pred) ** 0.5, 3),
            "R2": round(r2_score(y_test, pred), 3),
        })

    res = pd.DataFrame(results)
    best = res.sort_values("R2", ascending=False).iloc[0]
    print(res.to_string(index=False))
    print(f"\nBRIGHT best model: {best['model']}  (test R2 = {best['R2']})")

    best_model = tuned.get(best["model"], None)
    if best_model is None:
        for name, m in plain.items():
            if name == best["model"]:
                best_model = Pipeline([("sc", StandardScaler()), ("m", m)])
                best_model.fit(X_train, y_train)
    else:
        best_model = best_model.best_estimator_
        best_model.fit(X_train, y_train)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X_train.columns), r"C:\Users\IT SHOP\OneDrive\Documents\Default Project\model_features.pkl")
    print(f"saved model -> {MODEL_PATH}")

    demo = pd.DataFrame([{
        "Present_Price": 8.5, "Car_Age": 2026 - 2020, "Kms_Driven": 40000,
        "Owner": 0, "Fuel_Type_Petrol": 1, "Seller_Type_Individual": 0,
        "Transmission_Manual": 1,
    }])
    demo = demo.reindex(columns=X_train.columns, fill_value=0)
    print(f"new-car demo prediction (Present 8.5, 2020, 40k km, petrol, manual): "
          f"{float(best_model.predict(demo)[0]):.2f}")
    return res


baseline_df = baseline()
print(baseline_df.to_string(index=False))
improved()

print("\n========== COMPARISON (their notebook vs improved) ==========")
print("Their notebook (test): LR R2=0.75, DT R2=0.84  <- best, RF R2=0.52, GB R2=0.70")