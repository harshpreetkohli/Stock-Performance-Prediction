"""
This script is used to scrape the yahoo finanace api and transform
the data to be inserted to the DB
"""
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

SYMBOLS = {
    "MSFT": "Technology",
    "AAPL": "Technology",
    "NVDA": "Technology",
    "PFE": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "JPM": "Financial Services",
    "v": "Financial Services",
    "GS": "Financial Services",
    "XOM": "Energy",
    "NEE": "Energy",
    "CVX": "Energy"
    }


DB_PATH = "scripts/articles.db"
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)


def insert_stock_data(data: pd.DataFrame, sector: str,  cursor: sqlite3.Cursor) -> None:
    """
    Will Transform the data and 
    create the query to add the data in the DB
    """
    try:
        for _, row in data.iterrows():
            query = f"""
                    INSERT INTO stock_data (date, stock_symbol, sector, close_price) VALUES (
                    '{datetime.strftime(row.name, "%Y-%m-%d")}', '{row.keys()[0]}', '{sector}', {float(row.values[0])})
                """
            cursor.execute(query)
    except Exception as e:
        print(f"Error inserting stock data for {symbol}: {e}")


if __name__ == '__main__':
    for symbol, sector in SYMBOLS.items():
        data: pd.DataFrame = yf.download(symbol, START_DATE, END_DATE, interval='1d')
        data = data['Close']
        # Database Connection
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()

        insert_stock_data(data, sector, cursor)

        # Commit and Close the Database Connection
        connection.commit()
        cursor.close()
        connection.close()

    print("Data has been successfully inserted into the database.")