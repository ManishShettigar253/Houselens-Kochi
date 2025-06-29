import streamlit as st
import numpy as np
import pandas as pd

# ========== Model Preparation ==========
data = pd.read_csv("HPrice.csv")
data['Bedrooms'] = data['Bedrooms'].fillna(data['Bedrooms'].median())
data = data.dropna(subset=['Price'])

X = data[['Area', 'Bedrooms', 'Age']]
X.insert(0, 'B0', 1)
X = X.values
y = data['Price'].values

XT = X.T
XTX = XT.dot(X)
XTXINV = np.linalg.pinv(XTX)
XTy = XT.dot(y)
BHAT = XTXINV.dot(XTy)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Houselens Kochi", layout="centered", page_icon="🏠")

# Custom CSS and Navbar
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            width: 100%;
            font-weight: 600;
            background-color: #4CAF50;
            color: white;
            padding: 0.6em;
            border: none;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .result {
            background-color: #ffffff;
            padding: 1em;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            font-size: 18px;
        }
        .navbar {
            background-color: #ffffff;
            padding: 10px 20px;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .navbar h1 {
            font-size: 22px;
            color: #4CAF50;
            margin: 0;
        }
    </style>

    <div class="navbar">
        <h1>🏠 Houselens Kochi</h1>
    </div>
""", unsafe_allow_html=True)

# ======= App Body Title ========
st.markdown("<h2 style='text-align: center;'>Estimate the price of your flat, apartment or home in Kochi 💰</h2>", unsafe_allow_html=True)

# ======= Input Form ========
st.markdown("### 🧾 Enter Property Details")
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("📐 Area (in sqft)", min_value=300, max_value=10000, value=3000)

with col2:
    bedrooms = st.number_input("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3)

with col3:
    age = st.number_input("🏚️ Age of the Property (years)", min_value=0, max_value=100, value=10)

# ======= Predict Button ========
if st.button("📊 Predict Price"):
    price = BHAT[0] + BHAT[1]*area + BHAT[2]*bedrooms + BHAT[3]*age
    st.markdown(f"""
    <div class='result'>
        <strong>Estimated Price:</strong> ₹ {price:,.2f}
    </div>
    """, unsafe_allow_html=True)

# ======= Sample Data View ========
with st.expander("📂 View Sample Training Data"):
    st.dataframe(data.head())

# ========== Footer ==========
st.markdown("---", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Made with ❤️ by Manish</h5>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
    <a href="https://www.linkedin.com/in/manish-3b6142207/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=flat&logo=linkedin&logoColor=white" />
    </a>
    <a href="https://drive.google.com/file/d/19NSILJH3XAnF_TQn6vVmTZBMyGdkot8u/view?usp=drive_link" target="_blank">
        <img src="https://img.shields.io/badge/Resume-%23FF5733.svg?logo=document&logoColor=white" />
    </a>
    <a href="https://codolio.com/profile/ManishShettigar253" target="_blank">
        <img src="https://img.shields.io/badge/DSA_Stats-F28C28?logo=codeforces&logoColor=white&style=flat" />
    </a>
    <a href="https://manishshettigar253.github.io/Manish_Portfolio/" target="_blank">
        <img src="https://img.shields.io/badge/Portfolio-%230077B5.svg?logo=user&logoColor=white" />
    </a>
    <a href="https://www.youtube.com/@wanderTechEngineer" target="_blank">
        <img src="https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=youtube&logoColor=white" />
    </a>
    <a href="https://www.instagram.com/manish__shettigar?igsh=aGlwemQwdzc2N3g2" target="_blank">
        <img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=instagram&logoColor=white" />
    </a>
</div>
""", unsafe_allow_html=True)
