import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.utils import to_categorical
import yfinance as yf
import joblib
import yahooHistoricData
from datetime import datetime, timedelta

#load data
# call yahooHistoricData and pull data from yfinance
yahooHistoricData.dataPullYahoo()
data = pd.read_csv(r'C:\Users\yourusername\stockReportingSuite\data\yahooSemi_2010_2025.csv')

# Convert Close to numeric — this may introduce NaNs
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Define essential columns for your model
essential_columns = ['Close', 'SMA_10', 'RSI', 'VWAP']  # update this if needed

# Drop rows only if they are missing values in essential model features
data.dropna(subset=essential_columns, inplace=True)

# For other less critical columns (like P/E Ratio, Beta), fill missing values with their column means
non_essential_cols = ['P/E Ratio (Trailing)', 'P/E Ratio (Forward)', 'EPS (Trailing)',
                      'Dividend Yield', 'Beta']

for col in non_essential_cols:
    if col in data.columns:
        data[col].fillna(data[col].mean(), inplace=True)

# Define features and target
X = data[['Open', 'High', 'Low', 'Volume', 'Dividends', 'VWAP', 'SMA_10', 'RSI', 'Year', 'Month', 'Day', 'Weekday', 'Symbol',
          'P/E Ratio (Trailing)', 'P/E Ratio (Forward)', 'Market Cap', 'Dividend Yield', 'Beta']]
y = data['Close']

# Split data into train and test sets 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features/ group by each stock when training
cat_features = ['Symbol']
num_features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'VWAP', 'SMA_10', 'RSI', 'Year', 'Month', 'Day', 'Weekday',
                'P/E Ratio (Trailing)', 'P/E Ratio (Forward)', 'Market Cap', 'Dividend Yield', 'Beta']

# Preprocessing pipeline to train numerical and cat features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)

# Transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Reshape for Long Short-Term Model LSTM for temporal data analysis
X_train_processed = X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed
X_test_processed = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed

X_train_reshaped = X_train_processed.reshape((X_train_processed.shape[0], 1, X_train_processed.shape[1]))
X_test_reshaped = X_test_processed.reshape((X_test_processed.shape[0], 1, X_test_processed.shape[1]))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False), #hidden layers
    Dropout(0.2),
    Dense(25, activation='relu'), #use relu activation function for regression
    Dense(1)  # Single output for regression
])

# Compile Model with adam optimizer and MSE and MAE
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train Model
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

# Save Model
model.save('my_model.keras')
# Save preprocessing pipeline
joblib.dump(preprocessor, 'preprocessor.pkl')

# Evaluate on Test Data
test_loss, test_mae = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test Mean Absolute Error: {test_mae}")

# Predict on the test set
y_pred = model.predict(X_test_reshaped)

# Plot Actual vs Predicted ideal fit....................................
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred.flatten(), alpha=0.7, color='blue', edgecolor='k')
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--',
    linewidth=2,
    label='Ideal Fit'
)
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Actual Stock Prices')
plt.ylabel('Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.show()

# Plot Residuals Distribution.........................................
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Residual Errors')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#enter predicted closing stock values....................................
print("\n--- Predicted Closing Prices for Latest Available Date ---")

# Load your list of ETFs (Symbols) from the Excel file
df_etfs = pd.read_excel(r'C:\Users\yourusername\stockReportingSuite\data\Stocks.xlsx')
symbols = df_etfs['Stock ETF'].tolist()

#Start loop to step through ETF symbols based on number of symbols
for symbol in symbols:
    # Filter data for the symbol and get the latest date row
    symbol_data = data[data['Symbol'] == symbol]
    if symbol_data.empty:
        print(f"{symbol}: No data found in dataset.")
        continue

    latest_row = symbol_data.sort_values(by='Date', ascending=False).iloc[0]

    # Extract features for prediction - match exactly your features columns
    features = [
        'Open', 'High', 'Low', 'Volume', 'Dividends', 'VWAP', 'SMA_10', 'RSI',
        'Year', 'Month', 'Day', 'Weekday', 'Symbol',
        'P/E Ratio (Trailing)', 'P/E Ratio (Forward)', 'Market Cap', 'Dividend Yield', 'Beta'
    ]

    # Create a DataFrame with one row for the preprocessor
    X_pred = pd.DataFrame([latest_row[features]])

    # Fill missing values — safe strategy: 0 for Dividend, median for financial ratios
    X_pred.fillna({
        'P/E Ratio (Trailing)': X_train['P/E Ratio (Trailing)'].median(),
        'P/E Ratio (Forward)': X_train['P/E Ratio (Forward)'].median(),
        'Dividend Yield': 0,
        'Market Cap': X_train['Market Cap'].median(),
        'Beta': X_train['Beta'].median()
    }, inplace=True)

    # Optional: check again after fillna, skip if anything still missing
    if X_pred.isnull().any().any():
        print(f"{symbol}: Still missing values after imputation. Skipping prediction.")
        continue

    # Preprocess features
    X_pred_processed = preprocessor.transform(X_pred)
    if hasattr(X_pred_processed, 'toarray'):
        X_pred_processed = X_pred_processed.toarray()

    # Reshape for LSTM: (samples, timesteps, features)
    X_pred_reshaped = X_pred_processed.reshape((1, 1, X_pred_processed.shape[1]))

    # Predict closing price
    predicted_close = model.predict(X_pred_reshaped)[0][0]

    actual_close = latest_row['Close']
    print(f"{symbol}: Predicted Close = ${predicted_close:.2f} | Previous Close (latest) = ${actual_close:.2f}")