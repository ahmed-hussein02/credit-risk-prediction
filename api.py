from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create the FastAPI app
app = FastAPI(title="Credit Risk Prediction API")

# Test endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API is running"}


# Define the expected input data structure
class CustomerData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30to59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60to89DaysPastDueNotWorse: int
    NumberOfDependents: int


# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerData):
    # Convert input into a DataFrame matching the model's expected columns
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': customer.RevolvingUtilizationOfUnsecuredLines,
        'age': customer.age,
        'NumberOfTime30-59DaysPastDueNotWorse': customer.NumberOfTime30to59DaysPastDueNotWorse,
        'DebtRatio': customer.DebtRatio,
        'MonthlyIncome': customer.MonthlyIncome,
        'NumberOfOpenCreditLinesAndLoans': customer.NumberOfOpenCreditLinesAndLoans,
        'NumberOfTimes90DaysLate': customer.NumberOfTimes90DaysLate,
        'NumberRealEstateLoansOrLines': customer.NumberRealEstateLoansOrLines,
        'NumberOfTime60-89DaysPastDueNotWorse': customer.NumberOfTime60to89DaysPastDueNotWorse,
        'NumberOfDependents': customer.NumberOfDependents,
    }

    # Add engineered features (same logic as in the notebook)
    input_dict['WasIncomeMissing'] = 0
    input_dict['TotalTimesLate'] = (
        input_dict['NumberOfTime30-59DaysPastDueNotWorse'] +
        input_dict['NumberOfTime60-89DaysPastDueNotWorse'] +
        input_dict['NumberOfTimes90DaysLate']
    )
    input_dict['IncomePerDependent'] = input_dict['MonthlyIncome'] / (input_dict['NumberOfDependents'] + 1)

    input_df = pd.DataFrame([input_dict])

    # Predict probability of default
    probability = model.predict_proba(input_df)[0][1]
    prediction = "High Risk" if probability >= 0.5 else "Low Risk"

    return {
        "default_probability": round(float(probability), 4),
        "risk_level": prediction
    }