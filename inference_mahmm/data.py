import numpy as np
import pandas as pd 
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

#%%
class data:
    def __init__(self, code, date_start, date_fin_is, L, path):
        """
        Initializes the data class with the given parameters.
        
        Parameters
        ----------
        code : str
            The code of the index to retrieve.
        date_start : date
            The starting date of the data.
        date_fin_is : date
            The date marking the end of the training period (in-sample) and the start of the test period (out-of-sample).
        L : int
            The depth of the moving average.
        path : str, optional
            The path of the CSV file containing the data. 

        """
        
        self.path = path
        self.code = code
        self.L = L
        self.get(code, date_start)
        self.split_is_oos(date_fin_is)
   
    def moving_average(self, y):
        """
        Calculate the moving average of a given time series over a window of size 'L' using a sliding window approach

        Parameters
        ----------
        y : array-like
            The time series data.
    
        Returns
        -------
        numpy.ndarray
            An array containing the moving averages.
    
        """
        
        return np.array([y[t:t+self.L].sum()/self.L for t in range(len(y)-self.L)])  
   
    def compute_price(self, r, log = True): 
        """
        Compute the price series based on the given returns.
    
        Parameters
        ----------
        r : array-like
            The array of returns.
        log : bool, optional
            Determines whether to use logarithmic computation (default is True).
    
        Returns
        -------
        numpy.ndarray
            An array containing the computed price series.
    
        Notes
        -----
        This function initializes the price series at 100 and computes subsequent prices based on the provided returns.
        If 'log' is True, it uses logarithmic computation, otherwise it uses regular percentage returns.
    
        """

        price = np.ones(len(r)) * 100
        
        if log:
            for t in range(1, len(r)):
                price[t] = np.exp(np.log(price[t-1]) + r[t])  
        else:
            for t in range(1, len(r)):
                price[t] = price[t-1] * (1 + r[t])  
        return price
   
    def get(self, code, date_start):
        """
        Retrieve and preprocess the data for further analysis.
    
        Parameters
        ----------
        code : str
            The code of the index to retrieve.
        date_start : str
            The starting date for data retrieval (format: 'dd/mm/YYYY').
    
        Returns
        -------
        None
    
        Notes
        -----
        This method performs the following steps:
        1. Reads the data from the specified CSV file, selecting relevant columns.
        2. Converts the date column to datetime format.
        3. Filters the data to include only records after the specified start date.
        4. Computes percentage changes and log returns.
        5. Removes any NaN values.
        6. Sets class attributes for date, price, percentage changes, log returns, etc.
        7. Applies moving averages and computes adjusted price series.
    
        """
    
        df = pd.read_csv(self.path, delimiter=';', decimal=',') 
        df = df[['date', self.code]]
        df.columns = ['date', 'price']
        df.date = pd.to_datetime(df.date, dayfirst=True)
        
        self.eomy(df.date)
        df = df[df.date >= datetime.strptime(date_start, '%d/%m/%Y')].reset_index(drop=True)

        df['pct_change'] = df.price.pct_change(1) 
        df['log_ret'] =  (np.log(df.price.astype('float')) - np.log(df.price.shift(1).astype('float')))

        self.df = df.dropna(inplace=False).reset_index(drop=True)
        self.remove_outliers()
        self.date = self.df['date']
        self.price = np.array(self.df['price'])
        self.pct_change = np.array(self.df['pct_change'])
        self.log_ret = np.array(self.df['log_ret'])
        self.mm = self.moving_average(self.log_ret) 
        self.date_mm = self.date[self.L:]
        self.price = self.compute_price(self.log_ret)
        self.price_mm = self.compute_price(self.log_ret[self.L:])
        
    def split_is_oos(self, date_fin_is):
        """
        Split the data into in-sample (IS) and out-of-sample (OOS) sets based on a given date.
    
        Parameters
        ----------
        date_fin_is : date
            The date marking the end of the in-sample period.
    
        Returns
        -------
        None
    
        Notes
        -----
        This method divides the data into the following sets:
        - In-sample dates (date_is) and out-of-sample dates (date_oos).
        - In-sample percentage changes (pct_change_is) and out-of-sample percentage changes (pct_change_oos).
        - In-sample log returns (log_ret_is) and out-of-sample log returns (log_ret_oos).
        - In-sample computed prices (price_is) and out-of-sample computed prices (price_oos).
        - In-sample moving average adjusted prices (price_is_mm) and out-of-sample prices (price_oos_mm).
        - The proportion of data used for training (train_test_split).
    
        """
        
        self.date_is = self.date[self.date<date_fin_is].reset_index(drop=True)
        self.date_oos = self.date[self.date>=date_fin_is].reset_index(drop=True)
        self.date_is_mm, self.date_oos_mm = self.date_is[self.L:], self.date_oos
        
        self.pct_change_is = self.pct_change[self.date<date_fin_is]
        self.pct_change_oos = self.pct_change[self.date>=date_fin_is]
        
        self.log_ret_is = self.log_ret[self.date<date_fin_is]
        self.log_ret_oos = self.log_ret[self.date>=date_fin_is]
        
        self.price_is = self.compute_price(self.log_ret_is)
        self.price_oos = self.compute_price(self.log_ret_oos)
        
        self.price_is_mm = self.compute_price(self.log_ret_is[self.L:])
        self.price_oos_mm = self.price_oos
    
        self.train_test_split = len(self.date_is)/len(self.date)

    def eomy(self, dates):
        """
        Identify end-of-month and end-of-year dates.
    
        Parameters
        ----------
        dates : array-like
            An array containing date objects.
    
        Returns
        -------
        None
    
        Notes
        -----
        This method takes an array of date objects and performs the following steps:
        1. Creates lists of months (list_months) and unique years (list_years) from the provided dates.
        2. Constructs a 3D array (eom) where each element corresponds to the end-of-month date for a specific year and month combination.
        3. Extracts the end-of-year dates (eoy) from the eom array.
    
        """
        self.list_months = np.arange(1,13)
        self.list_years = np.unique([d.year for d in dates])[:-1]

        self.eom = np.array([[[d for d in dates if (d.month == m) & (d.year == y)][-1] for m in self.list_months if len([d for d in dates if (d.month == m) & (d.year == y)])>0] for y in self.list_years])
        self.eoy = np.array([e[-1] for e in self.eom])
        
    def remove_outliers(self, val_out=0.9):
        """
        Remove outliers from the dataset based on log returns.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
    
        Notes
        -----
        This method filters the dataset to include only records where log returns fall within the range (-0.9, 0.9).
        Any records outside this range are considered outliers and are removed.
    
        """
        
        self.df = self.df[(self.df['log_ret'] < val_out) & (self.df['log_ret'] > -val_out)].reset_index(drop=True)