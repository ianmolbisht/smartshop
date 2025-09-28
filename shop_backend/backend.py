# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "smartshop_model.pkl")
model = joblib.load(model_path)

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class Item(BaseModel):
    product_id: int
    day_num: int
    stock_left: int

# Root route for testing
@app.get("/")
def read_root():
    return {"message": "Backend is running"}

# Prediction route
@app.post("/predict")
def predict(item: Item):
    predicted_sales = int(model.predict([[item.product_id, item.day_num]])[0])
    
    if item.stock_left < 20:
        recommendation = f"Restock product {item.product_id}! Only {item.stock_left} left."
    elif predicted_sales > item.stock_left:
        recommendation = f"Order more product {item.product_id}. Predicted sales {predicted_sales}."
    else:
        recommendation = "Stock is sufficient."
    
    return {"predicted_sales": predicted_sales, "recommendation": recommendation}
