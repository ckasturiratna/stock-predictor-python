from fastapi import FastAPI
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

# ---------------------------------------------------------
# AI PRICE PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.get("/predict/{ticker}")
def predict(ticker: str):
    data = yf.download(ticker, period="6mo")["Close"]

    if len(data) < 20:
        return {"error": "Not enough data"}

    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values

    model = LinearRegression().fit(X, y)
    next_day = model.predict([[len(data) + 1]])

    return {
        "ticker": ticker.upper(),
        "predicted_price": float(next_day[0])
    }

# ---------------------------------------------------------
# HISTORY ENDPOINT (BUY DATE → TODAY)
# ---------------------------------------------------------
@app.get("/history/{ticker}")
def price_history(ticker: str, from_date: str):
    data = yf.download(ticker, start=from_date)

    if "Close" not in data:
        return {"error": "No data"}

    data.index = np.array(data.index, dtype="datetime64[D]")

    history = [
        {"date": idx.strftime("%Y-%m-%d"), "price": float(price)}
        for idx, price in zip(data.index, data["Close"].values)
    ]

    return {"ticker": ticker.upper(), "history": history}
