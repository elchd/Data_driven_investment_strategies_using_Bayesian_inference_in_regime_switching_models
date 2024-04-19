import numpy as np
import matplotlib.pyplot as plt
from prob_distrib import *
import tqdm.notebook as tq
from scipy.stats import norm
from scipy import stats
from scipy.special import logsumexp 
import random
import itertools
from collections import Counter
from pathlib import Path
import pickle
import os

class lma_hmm:
    def __init__(self, ν, Q, M, Σ, T, L, train_test_split=0.7):
        """
        Initialize the lma_hmm class with the given parameters.
    
        Parameters
        ----------
        ν : ndarray, shape (K,)
            The initial distribution of the hidden states.
        Q : ndarray, shape (K,K)
            The transition matrix of the hidden states.
        M : ndarray, shape (K,)
            The mean vector of the normal distribution for each state.
        Σ : ndarray, shape (K,)
            The standard deviation vector of the normal distribution for each state.
        T : int
            Total number of observations in the time series.
        L : int
            Depth of the moving average.
        train_test_split : float, optional
            Ratio of training data to total data. The default is 0.7.
    
        Notes
        -----
        This class initializes an LMA-HMM with specified parameters, including initial distribution, transition matrix,
        mean, and standard deviation of the normal distribution for each state. It also calculates the number of observations
        in the training set based on the specified train-test split ratio.
    
        """
        
        self.K = len(ν)
        self.T = T
        self.T_is = int(T*train_test_split)
        self.L = L
        self.true_ν = ν
        self.true_Q = Q
        self.true_M = M
        self.true_Σ = Σ
        self.get_combinations()
     
    def moving_average(self, y):
        """
        Calculate the moving average of a time series.
    
        Parameters
        ----------
        y : ndarray
            The input time series.
    
        Returns
        -------
        ndarray
            The moving average of the input time series.
    
        Notes
        -----
        This method computes the moving average of the input time series with a window size of 'L' if 'L' is not zero.
        If 'L' is zero, the original time series 'y' is returned.
    
        """
        
        if self.L==0:
            return y
        else:
            return np.array([y[t:(t+self.L+1)].sum()/(self.L+1) for t in range(len(y)-self.L)])
        
    def compute_z(self):
        """
        Compute the moving average of the time series of observations.
    
        Notes
        -----
        This method computes the moving average of the time series of observations using the 'moving_average' method.
        The result is stored in the 'true_z' attribute.
    
        """
       
        self.true_z = self.moving_average(self.true_y)
        
    def get_combinations(self, precalc = True):
        """
        Generate combinations and permutations of elements for the LMA-HMM model.
    
        Parameters
        ----------
        precalc : bool, optional
            Indicates whether to load precalculated combinations or compute them. The default is True.
    
        Returns
        -------
        None
    
        Notes
        -----
        This method generates combinations and permutations of elements for the LMA-HMM model.
        If 'precalc' is True, it loads precalculated combinations. Otherwise, it computes them and stores them in 'combi', 'permu', and 'combi_dict' attributes.

        """
        
        if precalc:
            with open(os.getcwd() + '/all_combi.pkl', 'rb') as f:
                x = pickle.load(f)
            self.combi = x[self.K][self.L][0]
            self.permu = x[self.K][self.L][1]
        else:
            elements = list(range(self.K))
            self.combi = list(itertools.combinations_with_replacement(elements, (self.L+1)))
            self.permu = list(itertools.product(elements, repeat=self.L+1))
        
    def simul(self, seed=123):
        """
        Simulate the true hidden state sequence and the observed output sequence using the LMA-HMM.
    
        Parameters
        ----------
        seed : int or None, optional
            Seed for random number generation. The default is 123.
    
        Notes
        -----
        This method generates hidden states using the initial distribution and transition matrix,
        and generates observations using normal distributions with means from 'true_M' and standard deviations from 'true_Σ'.
        The resulting time series is stored in 'true_y'. The moving average of 'true_y' is also computed and stored in 'true_z'.
        The data is then split into in-sample and out-of-sample sets.
    
        """
        
        x = np.zeros(self.T)
        if seed != None:
            np.random.seed(seed)
        x[0] = np.argmax(np.random.multinomial(1, self.true_ν))
        for t in range(1,self.T):
            x[t] = np.argmax(np.random.multinomial(1, self.true_Q[int(x[t-1])]))
        self.true_x = x
        
        if type(self.true_Σ)!=np.ndarray: 
            self.true_Σ = np.ones(self.K)*self.true_Σ
        self.true_y = np.array([np.random.normal(self.true_M[int(self.true_x[t])], self.true_Σ[int(self.true_x[t])]) for t in range(self.T)]) 
        
        self.compute_z()
        
        self.split_is_oos()
        
    def split_is_oos(self):
        """
        Split the simulated data into in-sample (IS) and out-of-sample (OOS) sets.
    
        Notes
        -----
        This method divides the simulated observations into in-sample and out-of-sample sets based on 'T_is' attribute.
        It also separates the corresponding hidden states and moving averages.
    
        """
    
        self.true_y_is, self.true_y_oos = self.true_y[:self.T_is], self.true_y[self.T_is:]
        self.true_x_is, self.true_x_oos = self.true_x[:self.T_is], self.true_x[self.T_is:]
        self.true_z_is, self.true_z_oos = self.true_z[:(self.T_is-self.L)], self.true_z[(self.T_is-self.L):]

    def G(self, x, y):
        """
        Compute emission function for the HMM.
    
        Parameters
        ----------
        x : int
            Hidden state.
        y : float
            Observation.
    
        Returns
        -------
        float
            Probability density function value.
    
        Notes
        -----
        This method computes the probability density function value for a given observation 'y' given the hidden state 'x'.
        It uses the mean and standard deviation of the normal distribution defined by the hidden state to calculate the PDF.
    
        """
        return pdf_normal(y, self.M[int(x)], self.Σ[int(x)])
    
    def G_ma(self, x, y):
        """
        Compute emission function for the LMA-HMM.
    
        Parameters
        ----------
        x : list
            List of hidden states.
        y : float
            Observation.
    
        Returns
        -------
        float
            Probability density function value.
    
        Notes
        -----
        This method computes the probability density function value for a given observation 'y' given the list of hidden states 'x'.
        It uses the mean and standard deviation of the normal distribution defined by the hidden states to calculate the PDF.
    
        """
        return pdf_normal(y, np.mean([self.M[int(i)] for i in x]), np.mean([self.Σ[int(i)] for i in x])/len(x))
    
    def true_G(self, x, y):
        """
        Compute emission function for the HMM.
    
        Parameters
        ----------
        x : int
            Hidden state.
        y : float
            Observation.
    
        Returns
        -------
        float
            Probability density function value.
    
        Notes
        -----
        This method computes the probability density function value for a given observation 'y' given the hidden state 'x'.
        It uses the true mean and true standard deviation of the normal distribution defined by the hidden state to calculate the PDF.
    
        """
        
        return pdf_normal(y, self.true_M[int(x)], self.true_Σ[int(x)])
    
    def true_G_ma(self, x, y):
        """
        Compute emission function for the LMA-HMM.
    
        Parameters
        ----------
        x : list
            List of hidden states.
        y : float
            Observation.
    
        Returns
        -------
        float
            Probability density function value.
    
        Notes
        -----
        This method computes the probability density function value for a given observation 'y' given the list of hidden states 'x'.
        It uses the true mean and true standard deviation of the normal distribution defined by the hidden states to calculate the PDF.
    
        """
        
        return pdf_normal(y, np.mean([self.true_M[int(i)] for i in x]), np.mean([self.true_Σ[int(i)] for i in x])/len(x))