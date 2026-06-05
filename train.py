import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lifetimes import BetaGeoFitter, GammaGammaFitter


# CONFIG
DATA_FILES = [
    "data/AliExpress.csv",
    "data/Ebay.csv",
    "data/lazada.csv",
    "data/walmart.csv"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENT_LABELS = ['Inactive', 'Requires Attension', 'Loyal Customers', 'Gold Members (Premiun)']


def load_csv(file):
    try:
        df = pd.read_csv(file, encoding="ISO-8859-1", low_memory=True,
                         dtype=str, on_bad_lines="skip")
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
        return df
    except Exception as e:
        print(f" CSV load failed: {file} → {e}")
        return None


EXACT_COL_MAP = {
    "date":     ["order_purchase_timestamp", "order_date", "invoicedate",
                 "date", "Order Date"],
    "price":    ["payment_value", "unit_price", "unitprice", "sales",
                 "price", "Sales", "Unit Price"],
    "customer": ["customer_unique_id", "customer_id", "customerid",
                 "client_id", "Customer ID"],
}


def detect(df, col_type, keywords):
    cols_lower = {c.lower(): c for c in df.columns}
    for exact in EXACT_COL_MAP.get(col_type, []):
        if exact.lower() in cols_lower:
            return cols_lower[exact.lower()]
    for col in df.columns:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None


def clean_data(df):
    df.columns = df.columns.str.strip()

    customer_col = detect(df, "customer", ["customer", "client"])
    date_col     = detect(df, "date",     ["date", "invoice", "purchase", "timestamp"])
    price_col    = detect(df, "price",    ["price", "sales", "amount", "value", "payment"])
    qty_col      = detect(df, "qty",      ["quantity", "qty", "ordered"])

    print(f"    customer={customer_col} | date={date_col} | price={price_col} | qty={qty_col}")

    if not date_col:
        print(" No date column — skipping")
        return None

    if not price_col:
        df["price_fallback"] = np.random.RandomState(42).randint(10, 500, len(df)).astype(str)
        price_col = "price_fallback"

    df["CustomerID"] = df[customer_col].astype(str) if customer_col else \
                       pd.Series(np.random.RandomState(42).randint(1, 500, len(df)).astype(str))
    df["OrderDate"]  = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col]    = pd.to_numeric(df[price_col], errors="coerce")
    df["Quantity"]   = pd.to_numeric(df[qty_col], errors="coerce").fillna(1) if qty_col else 1

    df = df.dropna(subset=["OrderDate", price_col])
    df["TotalPrice"] = df[price_col] * df["Quantity"]

    for col in ["TotalPrice"]:
        q1 = df[col].quantile(0.05)
        q2 = df[col].quantile(0.95)
        df[col] = np.clip(df[col], q1, q2)

    df = df[df["TotalPrice"] > 0]

    if len(df) > 30000:
        df = df.sample(30000, random_state=42)

    print(f"     Rows: {len(df)} | Avg TotalPrice: ${df['TotalPrice'].mean():.2f}")
    return df


def build_clv(df):
    import lifetimes
    clv = lifetimes.utils.summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="OrderDate",
        monetary_value_col="TotalPrice"
    )

    clv = clv[clv["frequency"] > 1]
    print(f"    Repeat customers: {len(clv)}")

    if len(clv) < 20:
        print("     Too few repeat customers → using all customers with frequency >= 1")
        clv = lifetimes.utils.summary_data_from_transaction_data(
            df,
            customer_id_col="CustomerID",
            datetime_col="OrderDate",
            monetary_value_col="TotalPrice"
        )
        clv = clv[clv["frequency"] >= 1]

    clv = clv[clv["monetary_value"] > 0]
    clv = clv.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"    CLV customers (monetary>0): {len(clv)}")
    return clv


def fit_bgf_ggf(clv):
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(clv["frequency"], clv["recency"], clv["T"])

    t = clv["T"].max()
    clv["expected_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, clv["frequency"], clv["recency"], clv["T"]
    )

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(clv["frequency"], clv["monetary_value"])

    clv["All-time"] = ggf.customer_lifetime_value(
        bgf,
        clv["frequency"],
        clv["recency"],
        clv["T"],
        clv["monetary_value"],
        time=t,
        freq="D",
        discount_rate=0.01
    )

    print(f"    All-time CLV range: ${clv['All-time'].min():.2f} → ${clv['All-time'].max():.2f} | mean: ${clv['All-time'].mean():.2f}")
    return clv


def train_models(clv):

    def assign_segment(clv_value):
        if clv_value < 500:
            return 'Inactive'
        elif clv_value < 2000:
            return 'Requires Attension'
        elif clv_value < 10000:
            return 'Loyal Customers'
        else:
            return 'Gold Members (Premiun)'

    clv["Segment"] = clv["All-time"].apply(assign_segment)

    print(" Segment distribution:")
    print(clv["Segment"].value_counts().to_string())

    clv_features = ["frequency", "recency", "T", "monetary_value"]
    X_clv = clv[clv_features]
    y_clv = clv["All-time"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clv, y_clv, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rf_model = RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=1
    )
    rf_model.fit(X_train_scaled, y_train)

    sample_preds = rf_model.predict(X_test_scaled[:3])
    print(f"    Sample CLV predictions: {np.round(sample_preds, 2)}")

    seg_features = ["frequency", "recency", "T", "monetary_value", "All-time"]
    X_seg = clv[seg_features]
    y_seg = clv["Segment"]

    seg_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=1
        ))
    ])
    seg_model.fit(X_seg, y_seg)

    return rf_model, seg_model, scaler, clv_features, seg_features


def save_models(name, rf_model, seg_model, scaler, clv_features, seg_features):
    joblib.dump(rf_model,     f"{MODEL_DIR}/{name}_clv.pkl")
    joblib.dump(seg_model,    f"{MODEL_DIR}/{name}_segment.pkl")
    joblib.dump(scaler,       f"{MODEL_DIR}/{name}_scaler.pkl")
    joblib.dump(clv_features, f"{MODEL_DIR}/{name}_clv_features.pkl")
    joblib.dump(seg_features, f"{MODEL_DIR}/{name}_seg_features.pkl")
    print(f" Saved models for: {name}")


if __name__ == "__main__":
    print(" STARTING TRAINING PIPELINE...\n")

    for file in DATA_FILES:
        print(f"\n{'='*30}\n {file}\n{'='*30}")

        try:
            df = load_csv(file)
            if df is None:
                continue

            df = clean_data(df)
            if df is None or len(df) == 0:
                print(" Skipping empty dataset")
                continue

            clv = build_clv(df)
            if clv is None or len(clv) < 20:
                print(" Not enough CLV data, skipping")
                continue

            clv = fit_bgf_ggf(clv)

            rf_model, seg_model, scaler, clv_features, seg_features = train_models(clv)

            name = os.path.basename(file).replace(".csv", "")
            save_models(name, rf_model, seg_model, scaler, clv_features, seg_features)

        except Exception as e:
            print(f" Error in {file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n ALL DATASETS PROCESSED")