import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# =========================
# CLEANING
# =========================
df["Income"] = df["Income"].fillna(df["Income"].median())
df.drop_duplicates(inplace=True)
df.columns = df.columns.str.strip()

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

# =========================
# FEATURE ENGINEERING
# =========================
df["Age"] = 2025 - df["Year_Birth"]

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
    df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

df["Customer_Year"] = df["Dt_Customer"].dt.year
df["Customer_Month"] = df["Dt_Customer"].dt.month

df.drop(["Dt_Customer", "ID"], axis=1, inplace=True, errors="ignore")

# =========================
# ENCODING
# =========================
df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT
# =========================
X = df.drop("Response", axis=1)
y = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# SAVE FILES ✅
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Files saved successfully!")