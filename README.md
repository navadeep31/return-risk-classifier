# E-Commerce Order Cancellation & Return Risk Intelligence System
## Multi-Class Order Risk Classifier (Low, Moderate, High Risk)

**Project Title:** Product Return and Cancellation Pattern Analysis-III  
**Category:** Machine Learning, FastAPI, Python, ngrok Deployment

---

## 🌐 Live Public Deployment & Swagger UI

You can test the deployed model in real time using the interactive Swagger UI or via any REST client like Postman:

| Service | Public URL | Description |
| :--- | :--- | :--- |
| 📄 **Interactive Swagger UI** | [https://sandpaper-cauterize-icky.ngrok-free.dev/docs](https://sandpaper-cauterize-icky.ngrok-free.dev/docs) | Test API directly from your browser |
| 🔮 **Prediction Endpoint** | `POST https://sandpaper-cauterize-icky.ngrok-free.dev/predict` | Main ML inference endpoint |
| ❤️ **Health Check** | `GET https://sandpaper-cauterize-icky.ngrok-free.dev/health` | Service status verification |

---

## 🧪 How to Test with Postman or Swagger UI

### Endpoint:
* **Method:** `POST`
* **URL:** `https://sandpaper-cauterize-icky.ngrok-free.dev/predict`
* **Headers:** `Content-Type: application/json`

### Sample Request Body (JSON):
```json
{
  "total_amount": 150.0,
  "total_items": 3,
  "is_guest": 0,
  "is_international": 1,
  "order_hour": 14
}
```

> **Note:** `average_price` is automatically computed internally as `total_amount / total_items` so the user does not have to provide redundant calculations.

### Expected API Response (JSON):
```json
{
  "risk_code": 0,
  "risk_level": "Low Risk",
  "recommended_action": "Auto-approve: Safe order for express dispatch.",
  "computed_average_price": 50.0
}
```

---

## 📁 Project Directory Structure
```text
return-risk-classifier/
├── order_risk_classification_m.ipynb   # Main Jupyter Notebook (Data prep, EDA, training)
├── backend.py                          # Production FastAPI API server
├── tunnel.py                           # ngrok public tunnel generator
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation and Swagger link
├── order_risk_model.pkl                # Trained Random Forest classifier
└── features.pkl                        # Serialized feature list
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server (Terminal 1)
```bash
uvicorn backend:app --reload --port 8000
```
Local Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Generate Public URL via ngrok (Terminal 2)
```bash
python tunnel.py
```
This generates the public HTTPS link forwarding to local port `8000`.

---

## 🎯 Model & Features

* **Algorithm:** Random Forest Classifier
* **Performance:** ~86.4% Accuracy, 0.83 Macro F1-Score
* **Classes:**
  * 🟢 **Low Risk (0):** Safe orders for automatic fulfillment.
  * 🟡 **Moderate Risk (1):** Orders requiring standard verification before dispatch.
  * 🔴 **High Risk (2):** Orders with elevated cancellation/return likelihood requiring manual hold.
