import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ─────────────────────────────
# 💰 Currency
# ─────────────────────────────
EUR_TO_INR = 90.5
def to_inr(x): return x * EUR_TO_INR

# ─────────────────────────────
# ⚙️ Page Config
# ─────────────────────────────
st.set_page_config(page_title="AI Laptop Predictor", layout="wide")

# ─────────────────────────────
# 🎨 UI STYLE
# ─────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
}
.hero-title {
    font-size:4rem;
    text-align:center;
    font-weight:800;
    background: linear-gradient(90deg,#00c6ff,#0072ff,#00ffcc);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.card {
    background: rgba(255,255,255,0.05);
    padding:25px;
    border-radius:15px;
    text-align:center;
}
.price {
    font-size:3rem;
    color:#00ff9f;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# 🏷️ TITLE
# ─────────────────────────────
st.markdown("<div class='hero-title'>💻 AI Laptop Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<center>Smart ML • Interactive • Advanced UI 🚀</center><br>", unsafe_allow_html=True)

# ─────────────────────────────
# 📊 DATA
# ─────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    companies = ['Apple','Dell','HP','Lenovo','Asus','MSI','Acer']
    types = ['Ultrabook','Notebook','Gaming']
    ram = ['4GB','8GB','16GB','32GB']

    data = []
    for i in range(600):
        c = np.random.choice(companies)
        t = np.random.choice(types)
        r = np.random.choice(ram)
        inches = np.random.uniform(13,17)
        weight = np.random.uniform(1.2,3)

        price = 500 + companies.index(c)*200 + types.index(t)*300 + int(r[:-2])*25
        price += np.random.normal(0,100)

        data.append([c,t,r,inches,weight,price])

    return pd.DataFrame(data,columns=['Company','Type','Ram','Inches','Weight','Price'])

df = generate_data()

# ─────────────────────────────
# 🎯 MODEL SELECTION
# ─────────────────────────────
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Decision Tree","Random Forest","Linear Regression"]
)

company = st.sidebar.selectbox("Company", df['Company'].unique())
typ = st.sidebar.selectbox("Type", df['Type'].unique())
ram = st.sidebar.selectbox("RAM", df['Ram'].unique())
inches = st.sidebar.slider("Screen Size",13.0,17.0,15.0)
weight = st.sidebar.slider("Weight",1.0,3.5,2.0)

# ─────────────────────────────
# 🤖 MODEL
# ─────────────────────────────
X = df.drop('Price',axis=1)
y = df['Price']

pre = ColumnTransformer([
    ('cat',OneHotEncoder(),['Company','Type','Ram']),
    ('num','passthrough',['Inches','Weight'])
])

if model_choice == "Decision Tree":
    reg = DecisionTreeRegressor(max_depth=5)
elif model_choice == "Random Forest":
    reg = RandomForestRegressor(n_estimators=100)
else:
    reg = LinearRegression()

model = Pipeline([('pre',pre),('reg',reg)])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test,preds)
mae = mean_absolute_error(y_test,preds)

# ─────────────────────────────
# 📊 METRICS
# ─────────────────────────────
col1,col2 = st.columns(2)

with col1:
    st.markdown(f"<div class='card'><h3>R² Score</h3><h2>{r2:.2f}</h2></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'><h3>MAE</h3><h2>₹{to_inr(mae):,.0f}</h2></div>", unsafe_allow_html=True)

# ─────────────────────────────
# 🔮 PREDICTION
# ─────────────────────────────
input_df = pd.DataFrame([[company,typ,ram,inches,weight]],
                        columns=['Company','Type','Ram','Inches','Weight'])

if st.sidebar.button("🚀 Predict Price"):
    with st.spinner("AI is thinking..."):
        time.sleep(1.5)
        prediction = model.predict(input_df)[0]

        price_inr = to_inr(prediction)

        # category
        if price_inr < 50000:
            category = "💸 Budget"
        elif price_inr < 100000:
            category = "⚖️ Mid-Range"
        else:
            category = "💎 Premium"

        st.markdown(f"""
        <div class='card'>
            <h3>Predicted Price</h3>
            <div class='price'>₹ {price_inr:,.0f}</div>
            <h4>{category} Laptop</h4>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"Model Used: {model_choice}")

# ─────────────────────────────
# 📊 INTERACTIVE GRAPH
# ─────────────────────────────
st.subheader("📊 Price Distribution")
fig = px.histogram(df, x=df['Price']*EUR_TO_INR, nbins=30)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# 🔥 FEATURE IMPORTANCE
# ─────────────────────────────
st.subheader("🔥 Feature Importance")

if hasattr(model.named_steps['reg'], "feature_importances_"):
    importances = model.named_steps['reg'].feature_importances_
    names = model.named_steps['pre'].get_feature_names_out()

    imp_df = pd.DataFrame({"Feature":names,"Importance":importances})
    imp_df = imp_df.sort_values(by="Importance",ascending=False).head(10)

    st.bar_chart(imp_df.set_index("Feature"))
else:
    st.info("Feature importance not available for Linear Regression")

# ─────────────────────────────
# 💡 INSIGHT
# ─────────────────────────────
st.info(f"""
💡 Insight:
A {ram} {typ} laptop from {company} is predicted as a **{model_choice}** outcome.
""")

# ─────────────────────────────
# 📥 DOWNLOAD
# ─────────────────────────────
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    "📥 Download Dataset",
    csv,
    "laptop_data.csv",
    "text/csv"
)

# ─────────────────────────────
# 📄 DATA
# ─────────────────────────────
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ─────────────────────────────
# FOOTER
# ─────────────────────────────
st.markdown("---")
st.markdown("<center>🚀 Advanced ML Project • Streamlit App</center>", unsafe_allow_html=True)