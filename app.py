import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
import io

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/hiteshisingla11-png/Customer-churn-analysis/main/churn_model.pkl"
    r = requests.get(url)
    return joblib.load(io.BytesIO(r.content))

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/hiteshisingla11-png/Customer-churn-analysis/main/customer_churn_cleaned.csv"
    return pd.read_csv(url)

@st.cache_data
def load_columns():
    url = "https://raw.githubusercontent.com/hiteshisingla11-png/Customer-churn-analysis/main/model_columns.pkl"
    r = requests.get(url)
    return joblib.load(io.BytesIO(r.content))

model = load_model()
df = load_data()
model_columns = load_columns()

st.title("📊 Customer Churn Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", "🎯 Risk Calculator", "📈 Feature Importance", "🔮 What-If Simulator"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['Exited'].mean()*100:.1f}%")
    col3.metric("Avg Age", f"{df['Age'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Probability Distribution")
        probs = model.predict_proba(df[model_columns])[:, 1]
        fig, ax = plt.subplots()
        ax.hist(probs, bins=30, color='#1D9E75')
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Number of customers")
        st.pyplot(fig)
    with col2:
        st.subheader("Age vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Exited', y='Age', ax=ax)
        st.pyplot(fig)

with tab2:
    st.subheader("Enter a customer's details to predict their churn risk")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 92, 35)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
    with col2:
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
    with col3:
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Active Member?", ["Yes", "No"])
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Calculate Churn Risk", type="primary"):
        input_dict = {
            'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': 1 if has_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': salary,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0,
            'Gender_Male': 1 if gender == "Male" else 0,
        }
        input_dict['BalanceSalaryRatio'] = balance / (salary + 1)
        input_dict['ProductDensity'] = num_products / (tenure + 1)
        input_dict['EngagementProduct'] = input_dict['IsActiveMember'] * num_products
        input_dict['AgeTenure'] = age * tenure

        input_df = pd.DataFrame([input_dict])[model_columns]
        prob = model.predict_proba(input_df)[0][1]

        st.metric("Predicted Churn Probability", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("⚠️ High churn risk — recommend retention action")
        else:
            st.success("✅ Low churn risk")

with tab3:
    st.subheader("Which factors matter most overall?")
    importance_df = pd.DataFrame({
        'Feature': model_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='#1D9E75')
    st.pyplot(fig)

with tab4:
    st.subheader("See how changing engagement affects churn risk")
    base_age = st.slider("Base Age", 18, 92, 40, key="sim_age")
    scenarios = []
    for active_state in [0, 1]:
        for products in range(1, 5):
            row = {
                'CreditScore': 650, 'Age': base_age, 'Tenure': 5, 'Balance': 50000,
                'NumOfProducts': products, 'HasCrCard': 1, 'IsActiveMember': active_state,
                'EstimatedSalary': 100000, 'Geography_Germany': 0,
                'Geography_Spain': 0, 'Gender_Male': 1
            }
            row['BalanceSalaryRatio'] = row['Balance'] / (row['EstimatedSalary']+1)
            row['ProductDensity'] = row['NumOfProducts'] / (row['Tenure']+1)
            row['EngagementProduct'] = row['IsActiveMember'] * row['NumOfProducts']
            row['AgeTenure'] = row['Age'] * row['Tenure']
            row_df = pd.DataFrame([row])[model_columns]
            prob = model.predict_proba(row_df)[0][1]
            scenarios.append({'Products': products, 'Active': 'Yes' if active_state else 'No', 'Risk': prob})
    sim_df = pd.DataFrame(scenarios)
    fig, ax = plt.subplots()
    sns.lineplot(data=sim_df, x='Products', y='Risk', hue='Active', marker='o', ax=ax)
    ax.set_ylabel("Predicted churn probability")
    st.pyplot(fig)
    st.caption("This shows how churn risk changes as a customer adopts more products.")
