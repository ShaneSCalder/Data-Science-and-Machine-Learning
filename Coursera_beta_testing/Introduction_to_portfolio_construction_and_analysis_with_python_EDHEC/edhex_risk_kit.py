import pandas as pd

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Returns a DataFrame with columns for:
    the wealth index,
    the previous peaks, and
    the percentage drawdown
    """
    
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({'Wealth': wealth_index,
                        'Previous Peak' : previous_peaks,
                        'Drawdown' : drawdowns})

def get_ffme_returns():
    
    '''
    Load the Fama-French Dataset for returns of the Top and Bottom Deciles by MarketCap
    '''
    portfolio = pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv',
                           header=0, index_col=0, na_values=-99.99)
    rets = portfolio[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(portfolio.index, format='%Y%m').to_period('M')
    return rets