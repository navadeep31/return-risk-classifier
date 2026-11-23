from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI app
app = FastAPI(title="Order Risk Classification API")

# 2. Load trained model and features
model = joblib.load("order_risk_model.pkl")
features = joblib.load("features.pkl")

# 3. Define Request Body
class Order(BaseModel):
    total_amount: float
    total_items: int
    unique_items: int
    average_price: float
    max_price: float
    is_bulk: int
    is_guest: int
    is_international: int
    order_hour: int
    is_weekend: int

# 4. Home endpoint
@app.get("/")
def home():
    return {"message": "Order Risk Classification API is running successfully!"}

# 5. Prediction endpoint
@app.post("/predict")
def predict_risk(order: Order):
    input_df = pd.DataFrame([order.dict()])[features]
    pred_code = int(model.predict(input_df)[0])

    labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    actions = {
        0: "Auto-approve: Safe order for express dispatch.",
        1: "Standard Review: Verify order details before shipping.",
        2: "Hold Order: High cancellation risk, require customer confirmation."
    }

    return {
        "risk_code": pred_code,
        "risk_level": labels[pred_code],
        "recommended_action": actions[pred_code]
    }
