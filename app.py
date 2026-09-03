import streamlit as st
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('xgb_model.pkl')

# Page config: sets the browser tab title
st.set_page_config(page_title="Credit Risk Prediction | Ahmed Hussein Nassar")

# Page title
st.title("Credit Risk Prediction")
st.caption("Built by Ahmed Hussein Nassar")
st.write("Enter customer details to predict their risk of default")


# Input fields for customer data
age = st.number_input("Age", min_value=18, max_value=100, value=35)
monthly_income = st.number_input("Monthly Income", min_value=0, value=3000)
debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3, step=0.1)
revolving_utilization = st.slider("Credit Utilization (0 = none, 1 = fully used)", 0.0, 1.5, 0.3)
open_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=5)
real_estate_loans = st.number_input("Number of Real Estate Loans", min_value=0, value=1)
dependents = st.number_input("Number of Dependents", min_value=0, value=0)

st.subheader("Payment History")
late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)


# Predict button
if st.button("Calculate Risk"):
    # Build the input dictionary matching the model's expected columns
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': dependents,
    }

    # Add the same engineered features used during training
    input_dict['WasIncomeMissing'] = 0
    input_dict['TotalTimesLate'] = late_30_59 + late_60_89 + late_90
    input_dict['IncomePerDependent'] = monthly_income / (dependents + 1)

    input_df = pd.DataFrame([input_dict])

    # Get prediction
    probability = model.predict_proba(input_df)[0][1]

    # Display result
    st.subheader("Result")
    st.write(f"Default Probability: {probability:.2%}")

    if probability >= 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px;">
        <a href="https://github.com/ahmed-hussein02" target="_blank" style="color: gray; text-decoration: none;">GitHub</a>
        &nbsp;•&nbsp;
        <a href="https://www.linkedin.com/in/ahmed-hussein-2a5422329" target="_blank" style="color: gray; text-decoration: none;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)