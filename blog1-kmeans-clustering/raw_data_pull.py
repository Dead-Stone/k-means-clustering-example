import pandas as pd
import yfinance as yf

def pull_stock_data(ticker, start_date="2006-01-01", end_date="2018-12-31"):
    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date).reset_index()
        stock_df.columns = [col.lower() for col in stock_df.columns]
        stock_df['ticker'] = ticker
        return stock_df
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return pd.DataFrame()

def main():
    tickers = pd.read_csv('./tickers.csv')
    ticker_list = tickers['ticker'].tolist()

    agg_df = pd.concat([pull_stock_data(tick) for tick in ticker_list], axis=0)
    agg_df.to_csv('./2006_2018_data.csv', index=False)

if __name__ == "__main__":
    main()
