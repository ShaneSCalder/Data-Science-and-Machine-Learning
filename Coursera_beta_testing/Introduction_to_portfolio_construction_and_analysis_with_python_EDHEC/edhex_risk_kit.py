import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm

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

def get_hifi_returns():
    
    '''
    Load and format EDHEC Hedge Fund Index Returns
    '''
    
    hfi = pd.read_csv('edhec-hedgefundindices.csv',
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    '''
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    '''

    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series of DataFrame
    Returns a float or a Series
    '''
    
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series of DataFrame
    Returns a float or a Series
    '''
    
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level default
    Returns True if the Hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level
    
def var_historic(r, level=5):
    
    """
    VaR Historic
    """
    
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected a Series of a DataFrame")  
        

def var_gaussian(r, level=5, modified=False):
    """
    returns the Parametric Gaussian Var of a Series or DataFrame
    If modified is True , then the modified Var is returned,
    using the Cornish_Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36)
    
    return -(r.mean() + z * r.std(ddof = 0))

def cvar_historic(r, level = 5):
    
    """
    Computes the Conditional VaR Series of DataFrame
    """
    
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else:
        raise TypeError("Expected a Series or a DataFrame")