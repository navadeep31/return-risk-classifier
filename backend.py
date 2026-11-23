from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI app
app = FastAPI(title="Order Risk Classification API")

# 2. Load trained model and features
model = joblib.load("order_risk_model.pkl")
features = joblib.load("features.pkl")

# 3. Define Request Body (Only 5 inputs required from the stakeholder!)
class Order(BaseModel):
    total_amount: float        # Total cart price in £
    total_items: int           # Total units in cart
    is_guest: int              # 1 if guest checkout, 0 if registered
    is_international: int      # 1 if shipping outside UK, 0 if domestic
    order_hour: int            # Hour of purchase (0 - 23)

# 4. Home endpoint
@app.get("/")
def home():
    return {"message": "Order Risk Classification API is running successfully!"}

# 5. Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Order Risk Classifier",
        "version": "1.0.0",
        "message": "API is up and running!"
    }

# 5. Prediction endpoint
@app.post("/predict")
def predict_risk(order: Order):
    # Automatically calculate average_price internally (don't trouble the user!)
    calculated_avg_price = round(order.total_amount / max(order.total_items, 1), 2)
    
    # Assemble complete feature dictionary for the model
    order_features = {
        "total_amount": order.total_amount,
        "total_items": order.total_items,
        "average_price": calculated_avg_price,
        "is_guest": order.is_guest,
        "is_international": order.is_international,
        "order_hour": order.order_hour
    }
    
    input_df = pd.DataFrame([order_features])[features]
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
        "recommended_action": actions[pred_code],
        "computed_average_price": calculated_avg_price
    }
