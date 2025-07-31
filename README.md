# Stock Portfolio Analysis & Machine Learning Predictor

A robust Python-based solution for managing personal stock portfolios, combining automated daily reporting and machine learning-based price prediction.
This suite has helped me personally track my investments and make informed decisions of when to sell and buy.
Using a Raspberry Pi as a server to run this software in combination with cron jobs has given me 
daily financial algorithmic and Machine Learning / AI based advice. This advice has helped me grow 
my personal investments with precise risk analysis.

---

## Table of Contents

- Overview
- Features
- Project Structure
- Getting Started
  - Prerequisites
  - Installation
  - Configuration
  - Usage
- Data Sources
- Technologies Used
- Future Enhancements
- License
- Contact

---

## Overview

This project automates the process of:

- Fetching up-to-date financial data
- Generating daily investment reports
- Predicting stock prices using an LSTM model

It enables users to track investments efficiently and make informed decisions through predictive analytics.

**Key skills demonstrated:**

- Python programming (OOP, scripting, automation)
- Data extraction and analysis (yfinance, pandas, scikit-learn)
- LSTM-based machine learning (TensorFlow/Keras)
- Email automation
- Data visualization (Matplotlib)
- Git version control

---

## Features

### Automated Daily Investment Reports

- Fetches real-time stock prices, volume, and RSI for holdings defined in `Stocks.xlsx`
- Calculates individual and total portfolio value
- Generates HTML reports with buy/sell/hold suggestions based on:
  - RSI (e.g., overbought/oversold) using 14 day window
  - Significant price change
- Emails reports to a specified recipient

### Historical Data Extraction

- Retrieves historical OHLCV and fundamental data via Yahoo Finance
- Calculates:
  - VWAP
  - SMA (10-day)
  - RSI
- Adds time-based features for ML (e.g., day, month, weekday, is_weekend)
- Saves all data into a CSV for training and prediction

### Stock Price Prediction with Machine Learning

- Implements an LSTM model to forecast closing prices
- Preprocessing includes:
  - StandardScaler (numerical features)
  - OneHotEncoder (categorical stock symbols)
- Includes:
  - Train/test split
  - Model training and evaluation
  - Visualization of predictions and residuals
- Saves:
  - Trained model (`my_model.keras`)
  - Preprocessing pipeline (`preprocessor.pkl`)
- Provides predicted closing prices for the latest data

---

## Project Structure

```
stockReportingSuite/
├── data
|    |__Stocks.xlsx
|    |__yahooSemi_2010_2025.csv
├── portfolio
|    |__ModelControl.py
├── Scripts
|    |__holdingsDailyReport.py
|    |__machineLearningPredictions.py
|         |__my_model.keras
|         |__preprocessor.pkl     
|    |__yahooHistoricData.py
├── tests
├── .gitignore
├── LICENSE.md
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

Clone the repository:
git clone https://github.com/cboyer-engineer/Stock-Portfolio-ML-Project
cd Stock-Portfolio-ML-Project
Create and activate a virtual environment:
Windows .venv/bin/activate
Linux python -m venv .venv

Dependencies install:
pip install -r requirements.txt
### Configuration
#### 1. Create `data/Stocks.xlsx`

A two-column Excel file with:

| Stock ETF | Shares |
|-----------|--------|
| AAPL      | 10     |
| MSFT      | 5      |
| GOOG      | 2      |

#### 2. Create a `.env` file in the root directory:
SENDER_EMAIL=your_email@gmail.com
EMAIL_APP_PASSWORD=your_gmail_app_password
RECIPIENT_EMAIL=recipient_email@example.com

> Use a Gmail App Password if 2FA is enabled.

#### 3. Update Paths (Optional)

Modify hardcoded paths in scripts to use relative paths, e.g. change:
r'C:\Users\YourName\Projects\stockReportingSuite\data\Stocks.xlsx'

## Usage

### 1. Extract Historical Data
python scripts/yahooHistoricData.py
This pulls and saves historical stock data into `data/yahooSemi_2010_2025.csv`.
### 2. Train and Predict with LSTM
python scripts/machineLearningPrediction.py
- Trains and evaluates an LSTM model
- Saves model and preprocessor
- Prints predictions
- Displays evaluation plots
### 3. Generate and Email Report
python scripts/holdingsDailyReport.py
- Fetches live prices
- Builds daily HTML report
- Emails it to recipient

---

## Data Sources

- **Stocks.xlsx**: Portfolio holdings
- **yahooSemi_2010_2025.csv**: Historical OHLCV + indicators + fundamentals
- **Yahoo Finance**: Via `yfinance` API

---

## Technologies Used

- Python
- Pandas
- NumPy
- yfinance
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- joblib
- smtplib + email.mime
- openpyxl
- python-dotenv

---

## Future Enhancements

- Interactive GUI (PyQt, Tkinter, Flask, or Dash)
- Automated scheduling (cron, Task Scheduler, APScheduler)
- Database integration (SQLite, PostgreSQL, MongoDB)
- Advanced models (ARIMA, Prophet, Transformers)
- Sentiment analysis integration
- Portfolio risk analysis (e.g., VaR, volatility)
- Expanded unit testing
- Cloud deployment (e.g., AWS Lambda)

---

## License

This project is licensed under the MIT License — see `LICENSE.md` for details.

---

## Contact

**Name:** Christopher Boyer  
**LinkedIn:** [linkedin.com/in/chris-boyer31415](https://linkedin.com/in/chris-boyer31415)  
**GitHub:** [cboyer-engineer](https://github.com/cboyer-engineer)

