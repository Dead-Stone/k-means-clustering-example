import pandas as pd

def calculate_returns_and_variance():
    agg_df = pd.read_csv('./2006_2018_data.csv')

    agg_df['year'] = pd.to_datetime(agg_df['date']).dt.year
    agg_df2 = agg_df.query("year >= 2012 and year <= 2017")

    first_dates = agg_df2.groupby(['ticker', 'year']).first().reset_index()
    last_dates = agg_df2.groupby(['ticker', 'year']).last().reset_index()

    returns = last_dates['close'] / first_dates['open'] - 1
    agg_df3 = first_dates[['ticker', 'year']].copy()
    agg_df3['returns'] = returns
    agg_df3 = agg_df3.groupby('ticker')['returns'].agg(['mean', 'var']).reset_index()
    agg_df3.columns = ['ticker', 'avg_yearly_returns', 'variance']

    agg_df3.to_csv('./processed_data.csv', index=False)

if __name__ == "__main__":
    calculate_returns_and_variance()
