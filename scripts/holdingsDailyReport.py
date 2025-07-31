import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

# Define your stock symbols and the number of shares you own from the Stock CSV into dictionary
df = pd.read_excel(r'C:\Users\Cdabo\PycharmProjects\stockReportingSuite\data\Stocks.xlsx')
holdings = dict(zip(df['Stock ETF'], df['Shares']))

# Fetch stock data
def fetch_stock_data(symbols):
    def calculate_rsi(symbol, period=14):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")  # Fetch 1 month of data
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None  # Return the latest RSI value

    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        rsi = calculate_rsi(symbol)
        data[symbol] = {
            "price": hist["Close"].iloc[-1],
            "change": hist["Close"].iloc[-1] - hist["Open"].iloc[-1],
            "percent_change": ((hist["Close"].iloc[-1] - hist["Open"].iloc[-1]) / hist["Open"].iloc[-1]) * 100,
            "volume": hist["Volume"].iloc[-1],
            "rsi": rsi
        }
    return data


# Generate the report
def generate_report(holdings, stock_data):
    total_portfolio_value = 0
    sell_recommendations = []
    table_data = []

    for symbol, shares in holdings.items():
        price = stock_data[symbol]["price"]
        change = stock_data[symbol]["change"]
        percent_change = stock_data[symbol]["percent_change"]
        volume = stock_data[symbol]["volume"]
        rsi = stock_data[symbol]["rsi"]
        value = price * shares
        total_portfolio_value += value

        # Realistic suggestion logic
        suggestion = "Hold"
        if percent_change < -5:
            suggestion = "Consider Selling (Significant Price Drop)"
            sell_recommendations.append(f"{symbol}: {percent_change:.2f}% drop")
        elif rsi is not None and rsi > 70:  # Add RSI > 70 sell suggestion
            suggestion = "Consider Selling (Overbought Conditions)"
            sell_recommendations.append(f"{symbol}: RSI is {rsi:.2f} (Overbought)")
        elif rsi is not None and rsi < 30:
            suggestion = "Consider Buying (Oversold)"

        table_data.append([symbol, shares, f"${price:.2f}", f"${change:.2f}", f"{percent_change:.2f}%", volume,
                           f"{rsi:.2f}" if rsi else "N/A", f"${value:.2f}", suggestion])

    report_html = f"""
    <html>
    <head><style>table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 5px; }}</style></head>
    <body>
    <h2>Daily Investment Report - {date.today()}</h2>
    {"".join(f"<p>{rec}</p>" for rec in sell_recommendations) if sell_recommendations else "<p>No stocks suggested to sell today.</p>"}
    <h3>Portfolio Summary</h3>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Shares</th>
            <th>Price</th>
            <th>Change</th>
            <th>% Change</th>
            <th>Volume</th>
            <th>RSI</th>
            <th>Value</th>
            <th>Suggestion</th>
        </tr>
        {"".join(f"<tr>{''.join(f'<td>{col}</td>' for col in row)}</tr>" for row in table_data)}
    </table>
    <p><strong>Total Portfolio Value:</strong> ${total_portfolio_value:.2f}</p>

    <!-- Add Criteria for Suggestions -->
    <p><strong>Criteria for Suggestions:</strong></p>
    <ul>
        <li><strong>Consider Selling:</strong> Triggered when the daily percent change is less than -5% or RSI is greater than 70.</li>
        <li><strong>Consider Buying:</strong> Triggered when the Relative Strength Index (RSI) is less than 30, indicating oversold conditions.</li>
        <li><strong>Hold:</strong> Suggested when neither of the above criteria is met.</li>
    </ul>

    </body>
    </html>
    """
    return report_html


# Send the email
def send_email(report_html, recipient_email):
    sender_email = "your_sending_email@gmail.com"
    app_password = "your_gmail_app_password"

    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "Daily Investment Report"

    msg.attach(MIMEText(report_html, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


if __name__ == "__main__":
    # Fetch stock data
    stock_data = fetch_stock_data(holdings.keys())

    # Generate the report
    report_html = generate_report(holdings, stock_data)

    # Send the report via email
    recipient_email = "your_recipient_email@example.com"
    send_email(report_html, recipient_email)
