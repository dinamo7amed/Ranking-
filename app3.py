import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

st.markdown("""
<style>
.stApp {
    background-color: #0B1F3A;
    color: white;
}

h1, h2, h3, p, div {
    color: white;
}

.stButton>button {
    background-color: #1F4E79;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("smart_parking_30k.csv")

users = {
    "ali": {"floor": 1, "type": "VIP"},
    "mona": {"floor": 2, "type": "normal"},
    "ahmed": {"floor": 3, "type": "disabled"}
}

def get_user(username):
    return users.get(username, {"floor": 1, "type": "normal"})

df['label'] = (
    (df['status'] == 0) &
    (df['distance'] < df['distance'].quantile(0.5))
).astype(int)

df_model = pd.get_dummies(df, columns=['type', 'time_period'], drop_first=True)

df_model = df_model.drop(['parking_id', 'spot_id'], axis=1)

X = df_model.drop(['label', 'user_id'], axis=1)
y = df_model['label']

model = lgb.LGBMClassifier()
model.fit(X, y)

st.title("🅿️ Smart Parking AI System")

username = st.text_input("Enter Username")

if username:

    user = get_user(username)

    st.success(f"Welcome {username}! ")

    preferred_floor = user["floor"]
    preferred_type = user["type"]

    if st.button("Refresh Parking Data 🔄"):
        random_idx = df.sample(frac=0.1).index
        df.loc[random_idx, 'status'] = np.random.choice([0,1], len(random_idx))

    available = df[df['status'] == 0].copy()

    available = pd.get_dummies(available, columns=['type','time_period'], drop_first=True)

    for col in X.columns:
        if col not in available.columns:
            available[col] = 0

    available = available[X.columns]

    available['score'] = model.predict_proba(available)[:, 1]

    top = available.sort_values('score', ascending=False).head(5)

    st.subheader("Top Recommended Parking Spots")
    st.dataframe(top)

    st.subheader(" Parking Map")

    fig, ax = plt.subplots()

    free = df[df['status'] == 0]
    occupied = df[df['status'] == 1]

    ax.scatter(free['x'], free['y'], c='green', label='Free')
    ax.scatter(occupied['x'], occupied['y'], c='red', label='Occupied')

    ax.set_facecolor("#0B1F3A")
    fig.patch.set_facecolor("#0B1F3A")

    ax.legend()
    st.pyplot(fig)

    st.subheader("Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Spots", len(df))
    col2.metric("Available", len(df[df['status']==0]))
    col3.metric("Occupied", len(df[df['status']==1]))

    st.bar_chart(df['floor'].value_counts())