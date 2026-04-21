# ==========================================

# 📊 MARKETING CAMPAIGN PREDICTION APP

# ==========================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ==========================================

# 1. LOAD DATA

# ==========================================

df = pd.read_csv("marketing_campaign.csv", sep='\t')

# ==========================================

# 2. CLEANING

# ==========================================

df['Income'] = df['Income'].fillna(df['Income'].median())
df.drop_duplicates(inplace=True)

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'], inplace=True, errors='ignore')

# ==========================================

# 3. FEATURE ENGINEERING

# ==========================================

df['Age'] = 2025 - df['Year_Birth']

df['Total_Spending'] = (
df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
)

df['Total_Children'] = df['Kidhome'] + df['Teenhome']

df['Total_Purchases'] = (
df['NumWebPurchases'] +
df['NumCatalogPurchases'] +
df['NumStorePurchases'] +
df['NumDealsPurchases']
)

df['Prev_Accept'] = (
df['AcceptedCmp1'] +
df['AcceptedCmp2'] +
df['AcceptedCmp3'] +
df['AcceptedCmp4'] +
df['AcceptedCmp5']
)

df['Customer_Days'] = (pd.Timestamp('2025-01-01') - df['Dt_Customer']).dt.days

df.drop(columns=['Year_Birth', 'Dt_Customer'], inplace=True)

# ==========================================

# 4. ENCODING

# ==========================================

df = pd.get_dummies(df, drop_first=True)

# ==========================================

# 5. SPLIT DATA

# ==========================================

X = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42,
stratify=y # ⭐ VERY IMPORTANT
)

# ==========================================

# 6. SCALING

# ==========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================

# 7. MODEL

# ==========================================

model = RandomForestClassifier(
n_estimators=300,
max_depth=10,
class_weight='balanced', # ⭐ handles imbalance
random_state=42
)

model.fit(X_train, y_train)

# ==========================================

# 8. PREDICTIONS

# ==========================================

y_prob = model.predict_proba(X_test)[:, 1]

# Adjust threshold here 👇

THRESHOLD = 0.25

y_pred = (y_prob > THRESHOLD).astype(int)

# ==========================================

# 9. EVALUATION

# ==========================================

print("\n=== MODEL PERFORMANCE ===")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================

# 10. STREAMLIT UI

# ==========================================

st.title("📊 Marketing Campaign Prediction")

tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Model", "💡 Insights"])

# ------------------ DASHBOARD ------------------

with tab1:
st.subheader("Customer Overview")

```
col1, col2, col3 = st.columns(3)
col1.metric("Customers", len(df))
col2.metric("Avg Income", int(df['Income'].mean()))
col3.metric("Avg Spending", int(df['Total_Spending'].mean()))

fig = px.histogram(df, x="Income", nbins=30, title="Income Distribution")
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(
    df,
    x="Income",
    y="Total_Spending",
    color="Response",
    title="Income vs Spending"
)
st.plotly_chart(fig, use_container_width=True)
```

# ------------------ MODEL ------------------

with tab2:
st.subheader("Model Performance")

```
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("AUC Score:", roc_auc_score(y_test, y_prob))

st.write("### Prediction Distribution")
st.bar_chart(pd.Series(y_pred).value_counts())
```

# ------------------ INSIGHTS ------------------

with tab3:
st.subheader("Key Insights")

```
st.write("✔ High income customers are more likely to respond")
st.write("✔ Customers with higher spending respond more")
st.write("✔ Previous campaign acceptance is a strong signal")
```

# ==========================================

# 11. SAMPLE PREDICTION

# ==========================================

st.subheader("🔍 Sample Prediction")

sample = X_test[0].reshape(1, -1)
prob = model.predict_proba(sample)[0][1]

if prob > THRESHOLD:
st.success(f"Likely to Respond ✅ ({prob:.2f})")
else:
st.warning(f"Less Likely ⚠️ ({prob:.2f})")
