import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64

# ✅ DEFINE FUNCTION FIRST
def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                            url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except:
        st.warning("⚠️ Background image not found.")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Clustering", page_icon="🚗", layout="wide")

# ✅ CALL FUNCTION AFTER DEFINITION
set_bg("car_bg.jpg")

# 👉 SET BACKGROUND IMAGE
set_bg("car_bg.jpg")

# ---------------- TITLE ----------------
st.title("🚗 Car Clustering (Unsupervised Learning)")
st.write("K-Means + PCA Visualization")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("car_price_prediction.csv")

df = load_data()

# ---------------- PREPROCESS ----------------
df = df.drop(['Car ID'], axis=1)

label_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

features = df[['Brand', 'Year', 'Engine Size', 'Fuel Type',
               'Transmission', 'Mileage', 'Condition', 'Price', 'Model']].dropna()

# ---------------- SCALING ----------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# ---------------- SLIDER ----------------
st.subheader("📊 Clustering Visualization")

k = st.slider("Select number of clusters (K)", 2, 6, 3)

# ---------------- K-MEANS ----------------
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

df = df.loc[features.index]
df['Cluster'] = clusters

# ---------------- PCA ----------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# ---------------- GRAPH (CENTERED) ----------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))

    scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'])

    ax.set_title("K-Means Clustering with PCA")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")

    st.pyplot(fig)

# ---------------- DATA VIEW ----------------
st.subheader("📄 Clustered Data")
st.dataframe(df.head(20))

# ---------------- DOWNLOAD ----------------
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Clustered Data",
    data=csv,
    file_name="clustered_cars.csv",
    mime="text/csv"
)
