import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def dataPullYahoo():
    # Pull stock symbol and date range from the Stock excel file
    df = pd.read_excel(r'C:\Users\yourusernames\stockReportingSuite\data\Stocks.xlsx')
    symbols = df['Stock ETF'].tolist()
    #define start date of stocks historical capture
    start_date = "2020-01-01"
    #define end date as current date or earlier date
    end_date = datetime.now() - timedelta(days=1)
    end_date = end_date.strftime("%Y-%m-%d")


    combined_data = pd.DataFrame()


    # Add Relative Strength Index (RSI)
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    for symbol in symbols:
        print(f"fetching data for {symbol}")
        try:
            # Fetch historical OHLCV data using yfinance
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            # Add VWAP (Volume Weighted Average Price)
            hist['VWAP'] = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()

            # Add Moving Average (SMA_10)
            hist['SMA_10'] = hist['Close'].rolling(window=10).mean()


            hist['RSI'] = calculate_rsi(hist['Close'])

            # Add Timestamp Features
            hist.reset_index(inplace=True)
            hist['Year'] = hist['Date'].dt.year
            hist['Month'] = hist['Date'].dt.month
            hist['Day'] = hist['Date'].dt.day
            hist['Weekday'] = hist['Date'].dt.weekday  # Numerical encoding for weekday (0=Monday, 6=Sunday)
            hist['Is_Weekend'] = hist['Date'].dt.weekday >= 5

            # Drop rows with NaN values introduced by rolling calculations
            hist = hist.dropna()

            # Fetch fundamental metrics from yfinance
            info = stock.info
            fundamental_data = {
                "Symbol": symbol,
                "P/E Ratio (Trailing)": info.get('trailingPE', None),
                "P/E Ratio (Forward)": info.get('forwardPE', None),
                "EPS (Trailing)": info.get('trailingEps', None),
                "Market Cap": info.get('marketCap', None),
                "Dividend Yield": info.get('dividendYield', None),
                "Beta": info.get('beta', None)
            }

             # Convert fundamental data to a DataFrame
            fundamentals_df = pd.DataFrame([fundamental_data])

            # Combine historical data with fundamentals
            # Merge historical data with fundamental data, duplicating rows for each date
            hist.reset_index(inplace=True)
            final_df = pd.concat([hist, pd.concat([fundamentals_df] * len(hist), ignore_index=True)], axis=1)
            # Append to the combined dataset
            combined_data = pd.concat([combined_data, final_df], ignore_index=True)

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    # Save to CSV
    output_file = r'C:\Users\yourusername\stockReportingSuite\data\yahooSemi_2010_2025.csv'
    combined_data.to_csv(output_file, index=False)
    #print(f"Dataset saved to {output_file}")
#un-comment to test if code works
#dataPullYahoo()
