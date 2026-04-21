# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# DUSTY ROSE × AMETHYST PALETTE
# #ede0e4  Blush (lightest pink)
# #dcc4ca  Dusty Rose
# #c9a4ae  Mauve
# #a87080  Rose Taupe (deep rose)
# #e8e2f0  Lavender Mist
# #c4b6d8  Soft Violet
# #a891c4  Amethyst Mid
# #8b6eb0  Amethyst
# #6b4e96  Deep Violet
# #52357c  Royal Plum
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

.stApp {
    background: linear-gradient(145deg,
        #f5edf0 0%,
        #ede0e4 10%,
        #dcc4ca 22%,
        #c9a4ae 36%,
        #d8cce8 50%,
        #c4b6d8 62%,
        #a891c4 75%,
        #8b6eb0 87%,
        #6b4e96 100%
    ) fixed !important;
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

.rose-header {
    background: linear-gradient(120deg,
        rgba(168,112,128,0.55) 0%,
        rgba(139,110,176,0.55) 60%,
        rgba(107,78,150,0.45) 100%
    );
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 22px;
    padding: 38px 44px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 12px 48px rgba(107,78,150,0.22), inset 0 1px 0 rgba(255,255,255,0.35);
}

.rose-header h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #2a1840;
    margin: 0 0 8px 0;
    letter-spacing: 0.5px;
    text-shadow: 0 1px 0 rgba(255,255,255,0.5);
}

.rose-header p {
    color: rgba(55,30,80,0.70);
    font-size: 0.92rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.8px;
}

section[data-testid="stSidebar"] {
    background: rgba(237,224,228,0.55) !important;
    backdrop-filter: blur(24px) saturate(1.3) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(200,164,174,0.35) !important;
}

section[data-testid="stSidebar"] * { color: #3a1e38 !important; }

section[data-testid="stSidebar"] h2 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #6b4e96 !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.38) !important;
    backdrop-filter: blur(18px) !important;
    border-radius: 16px !important;
    padding: 20px 18px !important;
    border: 1px solid rgba(196,182,216,0.45) !important;
    box-shadow: 0 4px 24px rgba(107,78,150,0.12), inset 0 1px 0 rgba(255,255,255,0.6) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 36px rgba(107,78,150,0.22) !important;
}

[data-testid="stMetricLabel"] {
    color: rgba(58,30,56,0.65) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #52357c !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.28);
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid rgba(196,182,216,0.35);
    backdrop-filter: blur(16px);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: rgba(82,53,124,0.65) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border: none !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="tab"] p { color: rgba(82,53,124,0.75) !important; font-weight: 600 !important; }

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #a87080, #8b6eb0) !important;
    color: white !important;
    box-shadow: 0 3px 14px rgba(139,110,176,0.35) !important;
}

.stTabs [aria-selected="true"] p { color: white !important; }

.stButton > button {
    background: linear-gradient(135deg, #a87080 0%, #8b6eb0 60%, #6b4e96 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    height: 52px !important;
    box-shadow: 0 4px 18px rgba(139,110,176,0.35) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 8px 28px rgba(107,78,150,0.50) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.08) !important;
}

input[type="number"] {
    background: rgba(255,255,255,0.45) !important;
    border: 1px solid rgba(196,182,216,0.5) !important;
    border-radius: 10px !important;
    color: #3a1e38 !important;
}

.glass-card {
    background: rgba(255,255,255,0.30);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(196,182,216,0.38);
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 6px 32px rgba(107,78,150,0.10), inset 0 1px 0 rgba(255,255,255,0.55);
    margin-bottom: 20px;
}

h3, h4 {
    color: #3a1e38 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 700 !important;
}

.stCode, code, pre {
    background: rgba(237,224,228,0.55) !important;
    border: 1px solid rgba(196,182,216,0.4) !important;
    border-radius: 10px !important;
    color: #52357c !important;
}

.stAlert {
    background: rgba(255,255,255,0.35) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(196,182,216,0.40) !important;
    color: #3a1e38 !important;
}

hr { border-color: rgba(196,182,216,0.30) !important; }

/* ─── SLIDER TRACK (background bar) ─── */
div[data-testid="stSlider"] div[role="slider"] ~ div,
div[data-baseweb="slider"] div[data-testid="stThumbValue"] ~ div,
div[data-baseweb="slider"] > div:first-child {
    background: rgba(196,182,216,0.35) !important;
    border-radius: 8px !important;
}

/* ─── SLIDER FILLED / ACTIVE RANGE ─── */
div[data-baseweb="slider"] [data-testid="stSlider"] div[style*="background"],
div[data-baseweb="slider"] > div > div[style*="background-color"] {
    background: linear-gradient(90deg, #a87080, #8b6eb0) !important;
}

/* ─── SLIDER THUMB KNOB ─── */
div[role="slider"] {
    background: linear-gradient(135deg, #a87080, #6b4e96) !important;
    border: 2px solid white !important;
    box-shadow: 0 2px 10px rgba(107,78,150,0.40) !important;
    width: 20px !important;
    height: 20px !important;
}

div[role="slider"]:focus {
    box-shadow: 0 0 0 4px rgba(139,110,176,0.30) !important;
}

/* ─── SLIDER VALUE TOOLTIP / THUMB LABEL ─── */
div[data-testid="stThumbValue"],
div[data-baseweb="tooltip"] div {
    background: #6b4e96 !important;
    color: white !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    padding: 2px 7px !important;
}

/* ─── SLIDER LABEL TEXT ─── */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p {
    color: #3a1e38 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
}

/* ─── SLIDER MIN/MAX VALUE TEXT ─── */
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: rgba(82,53,124,0.60) !important;
    font-size: 0.75rem !important;
}

/* ─── SELECTBOX TRIGGER (closed state) ─── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.45) !important;
    border: 1px solid rgba(196,182,216,0.5) !important;
    border-radius: 10px !important;
    color: #3a1e38 !important;
}

/* ─── SELECTBOX LABEL ─── */
.stSelectbox label, .stSelectbox p {
    color: #3a1e38 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ─── DROPDOWN POPOVER / LIST CONTAINER — full override ─── */
div[data-baseweb="popover"] { background: #f5edf0 !important; }
div[data-baseweb="popover"] > div { background: #f5edf0 !important; }
div[data-baseweb="select"] div[role="listbox"] { background: #f5edf0 !important; }
ul[role="listbox"] {
    background: #f5edf0 !important;
    border: 1px solid rgba(196,182,216,0.6) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(107,78,150,0.18) !important;
    padding: 6px !important;
}

/* ─── EVERY CHILD inside the popover ─── */
div[data-baseweb="popover"] *,
ul[role="listbox"] * {
    background-color: transparent !important;
    color: #3a1e38 !important;
}

/* ─── INDIVIDUAL OPTION ITEMS ─── */
li[role="option"] {
    background: transparent !important;
    color: #3a1e38 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    margin: 2px 0 !important;
}

/* ─── OPTION HOVER ─── */
li[role="option"]:hover {
    background: rgba(168,112,128,0.18) !important;
    color: #52357c !important;
}

/* ─── SELECTED / HIGHLIGHTED OPTION ─── */
li[role="option"][aria-selected="true"],
li[role="option"][data-highlighted="true"] {
    background: linear-gradient(90deg, rgba(168,112,128,0.20), rgba(139,110,176,0.20)) !important;
    color: #52357c !important;
    font-weight: 700 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(168,112,128,0.35); border-radius: 4px; }
</style>

<div class="rose-header">
    <h1>🛍️ Marketing Intelligence Dashboard</h1>
    <p>Consumer Behavior Analysis &nbsp;·&nbsp; Predictive Scoring &nbsp;·&nbsp; Campaign Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ───
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(237,224,228,0.72)',
    plot_bgcolor='rgba(255,255,255,0.18)',
    font=dict(color='#3a1e38', family='DM Sans, sans-serif'),
    title_font=dict(color='#52357c', size=15, family='Cormorant Garamond, serif'),
    xaxis=dict(gridcolor='rgba(139,110,176,0.10)', linecolor='rgba(139,110,176,0.20)'),
    yaxis=dict(gridcolor='rgba(139,110,176,0.10)', linecolor='rgba(139,110,176,0.20)'),
    margin=dict(t=55, b=40, l=30, r=30),
)
MIXED_SCALE  = [[0, '#dcc4ca'], [0.33, '#c9a4ae'], [0.66, '#a891c4'], [1, '#52357c']]
VIOLET_SCALE = [[0, '#e8e2f0'], [0.5, '#a891c4'], [1, '#52357c']]

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_campaign.csv", sep="\t")
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Age"] = 2026 - df["Year_Birth"]
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Customer_Year"]   = df["Dt_Customer"].dt.year
    df["Customer_Month"]  = df["Dt_Customer"].dt.month
    df["Customer_Tenure"] = (pd.to_datetime("today") - df["Dt_Customer"]).dt.days
    df.drop(["Dt_Customer", "ID"], axis=1, errors="ignore", inplace=True)
    return df

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("<h2>🎛️ Control Panel</h2>", unsafe_allow_html=True)
income_range    = st.sidebar.slider("Income Range", int(df["Income"].min()), int(df["Income"].max()), (int(df["Income"].min()), int(df["Income"].max())))
age_range       = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
spend_range     = st.sidebar.slider("Spending Range", int(df["Total_Spending"].min()), int(df["Total_Spending"].max()), (int(df["Total_Spending"].min()), int(df["Total_Spending"].max())))
response_filter = st.sidebar.selectbox("Response Filter", ["All", "1", "0"])

# =========================
# FILTER
# =========================
filtered_df = df[
    (df["Income"].between(*income_range)) &
    (df["Age"].between(*age_range)) &
    (df["Total_Spending"].between(*spend_range))
]
if response_filter == "1":   filtered_df = filtered_df[filtered_df["Response"] == 1]
elif response_filter == "0": filtered_df = filtered_df[filtered_df["Response"] == 0]

# =========================
# MODEL
# =========================
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Response", axis=1)
y = df_encoded["Response"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊  DASHBOARD", "🤖  MODEL PERFORMANCE", "🔮  PREDICTION ENGINE"])

# ─── TAB 1 ───
with tab1:
    st.markdown("### 📈 Key Business Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers",  len(filtered_df))
    c2.metric("💰 Avg Income",       f"₹{filtered_df['Income'].mean():,.0f}")
    c3.metric("🛒 Avg Spending",     f"₹{filtered_df['Total_Spending'].mean():,.0f}")
    c4.metric("🎯 Response Rate",    f"{filtered_df['Response'].mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig1 = px.histogram(filtered_df, x="Income", nbins=40, color_discrete_sequence=["#a87080"],
                        title="💰 Income Distribution of Customers")
    fig1.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig2 = px.scatter(filtered_df, x="Income", y="Total_Spending", color="Response",
                          color_discrete_map={0: "#c9a4ae", 1: "#6b4e96"},
                          opacity=0.80, size="Total_Spending",
                          title="💎 Income vs Spending Behavior")
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig3 = px.pie(filtered_df, names="Response",
                      color_discrete_sequence=["#6b4e96", "#c9a4ae"],
                      hole=0.50, title="🎯 Customer Response Distribution")
        fig3.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig4 = px.imshow(df.corr(numeric_only=True), text_auto=True,
                     color_continuous_scale=MIXED_SCALE, title="🔥 Feature Correlation Heatmap")
    fig4.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── TAB 2 ───
with tab2:
    st.markdown("### 🤖 ML Performance Breakdown")
    auc = roc_auc_score(y_test, y_prob)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("✅ Model Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    mc2.metric("📈 AUC Score",      f"{auc:.2%}")
    mc3.metric("🔁 Recall",         f"{recall_score(y_test, y_pred):.2%}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cm_fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True,
                           labels=dict(x='Predicted', y='Actual'),
                           x=['No Response', 'Response'], y=['No Response', 'Response'],
                           color_continuous_scale=VIOLET_SCALE, title="🔲 Confusion Matrix")
        cm_fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(15).sort_values()
        fig_fi = px.bar(feat_imp, orientation='h', color=feat_imp.values,
                        color_continuous_scale=MIXED_SCALE, title="🌳 Top 15 Feature Importances")
        fig_fi.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Classification Report")
    st.code(classification_report(y_test, y_pred))
    st.markdown('</div>', unsafe_allow_html=True)

# ─── TAB 3 ───
with tab3:
    st.markdown("## 🧠 AI Customer Scoring Panel")
    st.markdown("<p style='color:rgba(58,30,56,0.60);margin-bottom:24px;'>Enter customer details to evaluate campaign propensity.</p>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        income_input   = st.number_input("💰 Annual Income", value=int(df["Income"].median()))
        age_input      = st.number_input("👤 Customer Age", value=30)
    with col2:
        spending_input = st.number_input("🛒 Total Spending", value=500)
        days_input     = st.number_input("📅 Membership Days", value=365)
    threshold   = st.slider("⚙️ Model Sensitivity (Threshold)", 0.1, 0.9, 0.5)
    predict_btn = st.button("✦ Generate Intelligence Score")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame(columns=X.columns)
        input_df.loc[0] = 0
        if "Income"         in input_df.columns: input_df.loc[0, "Income"]         = income_input
        if "Age"            in input_df.columns: input_df.loc[0, "Age"]            = age_input
        if "Total_Spending" in input_df.columns: input_df.loc[0, "Total_Spending"] = spending_input

        prob  = model.predict_proba(scaler.transform(input_df))[0][1]
        score = int(prob * 100)

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            ring_color = "#6b4e96" if score >= 70 else "#a87080" if score >= 40 else "#c9a4ae"
            glow       = "rgba(107,78,150,0.25)" if score >= 70 else "rgba(168,112,128,0.25)" if score >= 40 else "rgba(201,164,174,0.20)"
            st.markdown(f"""
                <div style="text-align:center;padding:32px 20px;border:2px solid {ring_color};
                    border-radius:20px;background:rgba(255,255,255,0.32);backdrop-filter:blur(18px);
                    box-shadow:0 0 36px {glow},inset 0 1px 0 rgba(255,255,255,0.55);">
                    <div style="font-family:'Cormorant Garamond',serif;font-size:76px;color:{ring_color};
                        line-height:1;margin-bottom:8px;font-weight:700;">{score}</div>
                    <div style="color:rgba(82,53,124,0.65);font-size:0.72rem;font-weight:700;
                        letter-spacing:2.5px;text-transform:uppercase;">AI SCORE</div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if score >= 70:   st.success("### 🔥 High Value Target")
            elif score >= 40: st.warning("### ⚡ Nurture Segment")
            else:              st.error("### 🧊 Low Interest Segment")
            st.metric("Conversion Probability", f"{prob:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔍 AI Reasoning & Strategy")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Observation Factors**")
            if income_input   > df["Income"].median():         st.write("✅ High income segment detected.")
            if spending_input > df["Total_Spending"].median(): st.write("✅ High engagement historical data.")
            if age_input < 40:                                 st.write("✅ Fits target younger demographic.")
            st.markdown('</div>', unsafe_allow_html=True)
        with exp_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Actionable Strategy**")
            if score >= 70:   st.info("👉 Dispatch VIP Premium Catalog via Direct Mail.")
            elif score >= 40: st.info("👉 Target with Email Marketing + 10% Discount.")
            else:              st.info("👉 Hold Marketing Spend; Monitor for future activity.")
            st.markdown('</div>', unsafe_allow_html=True)