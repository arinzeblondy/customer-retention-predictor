import streamlit as st
import pandas as pd
import lightgbm as lgb
import pickle

# ✅ THIS MUST BE FIRST Streamlit command
st.set_page_config(page_title="Customer Retention Predictor", layout="centered")

# Load the model (you should replace this with the actual path to your saved model)
@st.cache_resource
def load_model():
    with open("lightgbm_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()


# Streamlit UI design

st.markdown("""
    <style>
        .main {
            background-color: #32174d;
            padding: 2rem;
            border-radius: 10px;
        }
        body {
            background-color: #800080;
            color: white;
            font-weight: bold;
        }
        .stApp {
            background-color: #800080;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
            font-weight: bold;
        }
        .css-1d391kg, .st-bb, .st-at {
            background-color: #ffffff0d !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📉 Customer Retention Predictor")
st.markdown("<p style='color: white; font-weight: bold;'>Upload a customer dataset and get churn predictions instantly.</p>", unsafe_allow_html=True)


st.warning("⚠️ Note: Your dataset must have exactly 10 feature columns. Please remove columns like 'CustomerID' or 'Churn' before uploading.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Feature encoding (ensure encoding matches training phase)
    cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]  # simplified encoding

    features = df.drop(columns=['CustomerID'], errors='ignore')

    # Ensure the input matches training features exactly
    features = df[[
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
        'Payment Delay', 'Subscription Type', 'Contract Length',
        'Total Spend', 'Last Interaction'
    ]]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    df['Predicted_Churn'] = predictions
    df['Churn_Probability'] = probabilities

    st.success("✅ Prediction Complete")
    st.write(df.head())

    st.download_button(
        label="📥 Download Results",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='churn_predictions.csv',
        mime='text/csv')

    st.markdown("---")
    st.markdown("<h3 style='color: white; font-weight: bold;'>📊 Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white;'>Predicted Churned Customers: <b>{int(df['Predicted_Churn'].sum())}</b></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white;'>Average Churn Probability: <b>{round(df['Churn_Probability'].mean() * 100, 2)}%</b></p>", unsafe_allow_html=True)
else:
    st.info("👈 Please upload a CSV file to begin.")
