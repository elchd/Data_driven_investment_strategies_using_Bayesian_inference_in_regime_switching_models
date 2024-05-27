from .ma_hmm import *
from .gibbs import *
from .data import *

def compute_price(r, log = True, nominal = 100): 
    """
    Compute the price series based on returns.

    Parameters
    ----------
    r : numpy.ndarray
        The time-series of returns.
    log : bool, optional
        If True, compute prices using logarithmic returns. If False, use arithmetic returns. 
        Defaults is True.
    nominal : float, optional
        The initial price (default is 100).

    Returns
    -------
    numpy.ndarray
        The time-series of computed prices.

    """
    price = np.ones(len(r)) * nominal
    
    if log:
        for t in range(1, len(r)):
            price[t] = np.exp(np.log(price[t-1]) + r[t])  
    else:
        for t in range(1, len(r)):
            price[t] = price[t-1] * (1 + r[t])  
    return price

def moving_average(y, n):
    """
    Compute the moving average of a time series.

    Parameters
    ----------
    y : numpy.ndarray
        The input time series data.
    n : int
        Window size for the moving average.

    Returns
    -------
    numpy.ndarray
        The time-series of the moving averages.

    """
    return(np.array([y[t:t+n+1].sum()/(n+1) for t in range(len(y)-n)]))

def compute_accuracy(x, true_x):
    """
    Compute the accuracy of predicted sequence compared to true sequence.

    Parameters
    ----------
    x : numpy.ndarray
        The predicted sequence.
    true_x : numpy.ndarray
        The true sequence.

    Returns
    -------
    accuracy : float
        Accuracy.

    """
    return len(x[x == true_x])/len(x)

def compute_shift_rate(x):
    """
    Compute the shift rate of a sequence.

    Parameters
    ----------
    x : numpy.ndarray
        The input sequence.

    Returns
    -------
    shift_rate : float
        Shift rate.

    """
    return(len(x[:-1][x[:-1] != x[1:]])/len(x[:-1]))

def compute_average_length_states(x):
    """
    Compute the average length of consecutive states in a sequence.

    Parameters
    ----------
    x : numpy.ndarray
        The input sequence of states.

    Returns
    -------
    average_length : float
        Average length of consecutive states.

    """
    
    state_lengths = []
    current_state = x[0]
    current_length = 1

    for i in range(1, len(x)):
        if x[i] != current_state:
            state_lengths.append(current_length)
            current_length = 1
            current_state = x[i]
        else:
            current_length += 1
    
    state_lengths.append(current_length)
    state_lengths = np.array(state_lengths)
    
    return np.mean(state_lengths)

def compute_back_and_forth_rate(x, n=1):
    """
    Compute the rate of back_and_forth shifts in a sequence.

    Parameters
    ----------
    x : numpy.ndarray
        Input sequence.
    n : int
        Length threshold for a state to be considered a false signal (default is 1).

    Returns
    -------
    false_signals_rate : float
        Rate of back_and_forth shifts.
        
    """
    
    state_counts = 0
    current_state = x[0]
    current_length = 1

    for i in range(1, len(x)):
        if x[i] != current_state:
            if current_length == n:
                state_counts += 1
            current_length = 1
            current_state = x[i]
        else:
            current_length += 1

    if current_length == n:
        state_counts += 1

    return state_counts/(len(x)-n)

def get_strat(x, fees, risky, riskfree, nominal, type_strat):
    """
    Return the time-series of prices and returns of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    fees : float
        The transaction cost rate.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    nominal : float
        The initial nominal value of the strategy.
    type_strat : str
        Type of strategy (binary or continuous).

    Returns
    -------
    prix_perf : numpy.ndarray
        The time-series of the trading strategy prices.
    perf_strat : numpy.ndarray
        The time-series of the trading strategy returns.

    """
    
    T = len(risky)
    if type_strat=='continuous':
        on_off = np.array(x).mean(axis=0)
        on_off = (on_off-on_off.min())/(on_off.max()-on_off.min())
    else:
        on_off = np.array(x)/np.max(x)
    on_off = np.append(on_off[0], on_off[:-1])

    fees_list = np.zeros(T)
    fees_list[1:] = fees * abs(on_off[:-1] - on_off[1:])
        
    perf_strat = ((1 + ((on_off * risky[-T:]) + ((1 - on_off) * riskfree[-T:]))) * (1 - fees_list) - 1)
   
    prix_perf = [nominal]
    for t in range(1,len(perf_strat)):
        prix_perf.append(prix_perf[-1]*np.exp(perf_strat[t]))
    prix_perf = np.array(prix_perf)

    return (prix_perf, perf_strat)

def get_fees_strat(x, fees, risky, type_strat):
    """
    Compute the total transaction fees incurred by a trading strategy.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    fees : float
        The transaction cost rate.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous).

    Returns
    -------
    float
        Total transaction fees incurred by the strategy.
    """
    
    T = len(risky)
    if type_strat=='continuous':
        on_off = np.array(x).mean(axis=0)
        on_off = (on_off-on_off.min())/(on_off.max()-on_off.min())
    else:
        on_off = np.array(x)/np.max(x)
    on_off = np.append(on_off[0], on_off[:-1])

    fees_list = np.zeros(T)
    fees_list[1:] = fees * abs(on_off[:-1] - on_off[1:])
    
    return np.sum(fees_list)

def compute_perf_strat(x, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the performance of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The performance of the strategy.

    """
    strat, _ = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    return (strat[-1] / strat[0] - 1)*100

def compute_perf_strat_yearly(x, date, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the annual performances of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The annual performances of the strategy.

    """
    strat, _ = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    strat = np.array(strat)
    perf_annual = []
    year_end = date[0]
    list_years = np.unique([d.year for d in date])
    for i in range(len(list_years)-1):
        year_start = year_end
        if i == len(list_years)-1:
            year_end = date.iloc[-1]
        else:
            year_end = date.loc[date<=str(list_years[i+1])+'-12-31'].iloc[-1]
        perf_annual.append(float(strat[date == year_end]/strat[date == year_start]-1)*100)
        
    return perf_annual

def compute_avg_vol_strat(x, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the volatility of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The volatility of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    return (np.std(perf) * np.sqrt(252))*100

def compute_avg_vol_strat_yearly(x, date, risky, riskfree, type_strat = 'binary', fees=0.001, nominal=100):
    """
    Return the annual volatilities of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The annual volatilities of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    vol_annual = []
    year_end = date[0]
    list_years = np.unique([d.year for d in date])

    for i in range(len(list_years)-1):
        year_start = year_end
        if i == len(list_years)-2:
            year_end = date.iloc[-1]
        else:
            year_end = date.loc[date<=str(list_years[i+1])+'-12-31'].iloc[-1]
        vol_annual.append(np.std(perf[(date >= year_start)&(date <= year_end)])*np.sqrt(252)*100)
        
    return vol_annual

def compute_avg_vol_strat_monthly(perf, date):
    """
    Return the monthly volatilities of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The monthly volatilities of the strategy.

    """
    vol_monthly = []
    year_month_end = str(date[0].year) + '-' + str(date[0].month).zfill(2)

    for i in range(len(date)-1):
        year_month_start = year_month_end
        if i == len(date)-2:
            year_month_end = str(date.iloc[-1].year) + '-' + str(date.iloc[-1].month).zfill(2)
        else:
            year_month_end = str(date.iloc[i+1].year) + '-' + str(date.iloc[i+1].month).zfill(2)
        if year_month_start != year_month_end:
            vol_monthly.append(np.std(perf[(date >= year_month_start) & (date < year_month_end)]))
        
    return vol_monthly

def compute_mdd_strat(x, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the maximum draw-down of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The maximum draw-down of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    cum_perf = pd.Series((1 + perf).cumprod())
    return (cum_perf.cummax() - cum_perf).max()

def compute_mdd_strat_yearly(x, date, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the annual maximum draw-downs of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The annual maximum draw-downs of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    mdd_annual = []
    year_end = date[0]
    list_years = np.unique([d.year for d in date])

    for i in range(len(list_years)-1):
        year_start = year_end
        if i == len(list_years)-2:
            year_end = date.iloc[-1]
        else:
            year_end = date.loc[date<=str(list_years[i+1])+'-12-31'].iloc[-1]
        cum_perf = pd.Series((1 + perf[(date >= year_start)&(date <= year_end)]).cumprod())
        mdd_annual.append((cum_perf.cummax() - cum_perf).max())
        
    return mdd_annual

def compute_sr_strat(x, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the Sharpe ratio of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The Sharpe ratio of the strategy.

    """
    return compute_perf_strat(x, risky, riskfree, type_strat, fees, nominal) / compute_avg_vol_strat(x, risky, riskfree, type_strat, fees, nominal)

def compute_sr_strat_yearly(x, date, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the annual Sharpe ratios of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The annual Sharpe ratios of the strategy.

    """
    perf = compute_perf_strat_yearly(x, date, risky, riskfree, type_strat, fees, nominal)
    vol = compute_avg_vol_strat_yearly(x, date, risky, riskfree, type_strat, fees, nominal)
    return np.array(perf)/np.array(vol)

def compute_ir_strat(x, risky, riskfree, type_strat = 'binary', fees=0.001, nominal=100):
    """
    Return the information ratio of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The information ratio of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    return np.mean(perf - risky[-len(perf):])/np.std(perf - risky[-len(perf):])

def compute_ir_strat_yearly(x, date, risky, riskfree, type_strat='binary', fees=0.001, nominal=100):
    """
    Return the annual information ratios of a trading strategy based on hidden states signal.

    Parameters
    ----------
    x : numpy.ndarray
        The time-series of hidden states.
    date : numpy.ndarray
        The series of dates.
    risky : numpy.ndarray
        The time-series of a risky asset log-returns.
    riskfree : numpy.ndarray
        The time-series of a risk-free asset log-returns.
    type_strat : str
        Type of strategy (binary or continuous). Default is binary.
    fees : float
        The transaction cost rate (default is 0.1%).
    nominal : float
        The initial nominal value of the strategy (default is 100).

    Returns
    -------
    float
        The annual information ratios of the strategy.

    """
    _, perf = get_strat(x, fees, risky, riskfree, nominal, type_strat)
    ir_annual = []
    year_end = date[0]
    list_years = np.unique([d.year for d in date])

    for i in range(len(list_years)-1):
        year_start = year_end
        if i == len(list_years)-2:
            year_end = date.iloc[-1]
        else:
            year_end = date.loc[date<=str(list_years[i+1])+'-12-31'].iloc[-1]
        if np.std(perf[(date >= year_start)&(date <= year_end)] - risky[(date >= year_start)&(date <= year_end)]) != 0:
            ir_annual.append(np.mean(perf[(date >= year_start)&(date <= year_end)] - risky[(date >= year_start)&(date <= year_end)])/np.std(perf[(date >= year_start)&(date <= year_end)] - risky[(date >= year_start)&(date <= year_end)]))
        else:
            ir_annual.append(0)
        
    return ir_annual
    
def run_gibbs_yearly(K, n, DATA, niter=15, nburnin=5):
    list_eoy = np.append(np.append([datetime.strptime(date_start, '%d/%m/%Y')],DATA.last_day_year), [DATA.date[len(DATA.date)-1]])
    x_is, x_oos = [], []
    x_all_is, x_all_oos = np.ones((niter-nburnin-1, len(DATA.date[DATA.date<=list_eoy[-2]])))*3, np.ones((niter-nburnin-1, len(DATA.date[(DATA.date>list_eoy[-11])])))*3
    
    last_index_oos = 0
    for i in range(11, 1, -1): 
        log_ret_is = DATA.log_ret[(DATA.date<=list_eoy[i-4]) & (DATA.date<=list_eoy[i])]
        log_ret_oos = DATA.log_ret[(DATA.date>list_eoy[i]) & (DATA.date<=list_eoy[i+1])]
        log_ret = np.append(log_ret_is, log_ret_oos)
                
        GIBBS = gibbs(K, n, len(log_ret_is)/len(log_ret), log_ret)
        GIBBS.algorithm_est_ma(niter, nburnin)
        
        if x_is == []:
            x_is = np.concatenate([x_is, GIBBS.x_is])
            
            for j, k in enumerate(GIBBS.res.keys()):
                x_all_is[j][:len(GIBBS.res[k]['x_is'])] = GIBBS.res[k]['x_is']
                x_all_oos[j][:len(GIBBS.res[k]['x_oos'])] = GIBBS.res[k]['x_oos']
                
        else:
            x_is = np.concatenate([x_is, GIBBS.x_is[-(len(log_ret_is)-len(x_is)):]])
            
            for j, k in enumerate(GIBBS.res.keys()):
                x_all_is[j][len(x_is)-len(GIBBS.x_is):len(GIBBS.x_is)] = GIBBS.res[k]['x_is'][-(len(x_is)-len(GIBBS.x_is)):]
                x_all_oos[j][last_index_oos:last_index_oos+len(GIBBS.x_oos)] = GIBBS.res[k]['x_oos']
                
        last_index_oos += len(GIBBS.x_oos)
        
        x_oos = np.concatenate([x_oos, GIBBS.x_oos])
    
    return(x_is, x_oos, x_all_is, x_all_oos)