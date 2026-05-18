mport streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="飯店預訂取消預測儀表板", layout="wide")

@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('hotel_bookings.csv')
    df.loc[df['adr'] <= 0, 'adr'] = df['adr'].median()
    df.loc[df['adults'] == 0, 'adults'] = df['adults'].mode()[0]
    df['children'] = df['children'].fillna(0)
    
    features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                'adults', 'previous_cancellations', 'adr']
    X = df[features].dropna()
    y = df.loc[X.index, 'is_canceled']
    return X, y, features

X, y, features = load_and_prep_data()

# 模型訓練
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

st.title("🏨 飯店預訂狀態預測系統 (Naive Bayes)")
st.write("本儀表板基於單純貝氏模型，即時預測訂單為「已入住」或「取消」。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("設定訂單參數")
    lead_time = st.slider("預訂前置時間 (天)", 0, 500, 100)
    adr = st.slider("平均日房價 (ADR)", 50, 500, 100)
    adults = st.number_input("成人数", 1, 10, 2)
    
with col2:
    st.subheader("歷史紀錄")
    prev_cancel = st.number_input("過去取消次數", 0, 10, 0)
    week_nights = st.slider("平日住宿晚數", 0, 10, 2)
    weekend_nights = st.slider("假日住宿晚數", 0, 10, 1)

# 預測
input_data = pd.DataFrame([[lead_time, weekend_nights, week_nights, adults, prev_cancel, adr]], columns=features)
input_scaled = scaler.transform(input_data)
pred = gnb.predict(input_scaled)
prob = gnb.predict_proba(input_scaled)

st.markdown("---")
st.subheader("預測結果")
if pred[0] == 1:
    st.error(f"⚠️ 高風險！此訂單預測為：**取消 (Canceled)** (機率: {prob[0][1]:.2%})")
else:
    st.success(f"✅ 安全！此訂單預測為：**已入住 (Check-Out)** (機率: {prob[0][0]:.2%})")