# Stock Market Prediction App — Next 30 Days

This Streamlit app allows you to upload a historical stock price CSV, choose your model (Holt-Winters or Prophet), and forecast the next 30 days (or custom horizon) of prices.

## Features
- CSV upload or demo data
- Choose frequency (daily, weekly, etc.)
- Choose model (Holt-Winters or Prophet)
- 30-day interactive forecast plot
- Downloadable forecast CSV

## Installation
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

> Note: Prophet may require additional system dependencies to install.
