import numpy as np
from .random_sample import *
from .ma_hmm import *
import tqdm.notebook as tq
import warnings


class gibbs(ma_hmm):
    def __init__(self, K, L, train_test_split = 0.7, y = None, index_is_oos=None):
        """
        Initialize the Gibbs sampler class for LMA-HMM with the given parameters.
        
        Parameters
        ----------
        K : int
            The number of hidden states.
        L : int
            The depth of the moving average.
        train_test_split : float, optional
            The proportion of data to use for training (default is 0.7).
        y : numpy.ndarray, optional
            The input time series data (default is None).
        index_is_oos : int, optional
            Index marking the separation between in-sample and out-of-sample data (default is None).

        Notes
        -----
        This class extends the lma_hmm class for Gibbs sampling specific functionality. 
        It initializes with parameters for the number of states (K), moving average depth (L), 
        training-test split ratio, input data (y), and the index separating in-sample and out-of-sample data.

        """

        self.K = K
        self.L = L
        self.tts = train_test_split
        if type(y) == np.ndarray:
            if index_is_oos == None:
                self.true_y, self.true_y_is, self.true_y_oos = y, y[:int(len(y)*self.tts)], y[int(len(y)*self.tts):]   
            else:
                self.true_y, self.true_y_is, self.true_y_oos = y, y[:index_is_oos], y[index_is_oos:]
            
            self.true_z = self.moving_average(self.true_y)
            self.true_z_is, self.true_z_oos = self.true_z[:(len(self.true_y_is)-self.L)], self.true_z[(len(self.true_y_is)-self.L):]    
            self.T, self.T_is, self.T_oos = len(self.true_y), len(self.true_y_is), len(self.true_y_oos)
            self.get_combinations()
            self.initialize() 
        
    def initialize(self):
        """
        Initialize the hyperparameters and priors for Gibbs sampling.
    
        Notes
        -----
        This method sets initial values for the hyperparameters ξ, κ, α, g, and h based on properties of the in-sample data.
        It also computes the corresponding priors needed for the Gibbs sampling algorithm.
    
        """
        
        R = self.true_y_is.max() - self.true_y_is.min()
        self.ξ = (self.true_y_is.min() + self.true_y_is.max())/2
        self.κ = 1/(R**2)
        self.α = 2
        self.g = 0.2
        self.h = 10/(R**2)
        self.compute_priors()
    
    def pretrain(self, ν, Q, M, Σ, T):
        """
        Perform pretraining using specified parameters.

        Parameters
        ----------
        ν : ndarray, shape (K,)
            The initial distribution vector of the hidden states.
        Q : ndarray, shape (K,K)
            The transition matrix of the hidden states.
        M : ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.
        Σ : ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
        T : int
            The length of the time series to be simulated.
    
        Notes
        -----
        This method initializes the LMA-HMM with specified parameters and performs pretraining to estimate initial parameters.
        It first sets up the LMA-HMM instance with provided parameters (ν, Q, M, Σ, T) and simulates hidden states and observations.
        Then, it initializes the parameters for Gibbs sampling.
            
        """
        
        super().__init__(ν, Q, M, Σ, T, self.L, self.tts)
        super().simul()
        self.initialize() 
               
    def compute_priors(self):
        """
        Compute the prior parameters for the LMA-HMM:
            - Dirichlet prior for ν
            - Dirichlet prior for Q
            - Normal prior for M
            - Gamma prior for β
            - Inverse-Gamma prior for Σ
    
        """
        
        self.ν = np.random.dirichlet(np.ones(self.K))
        
        mat = np.ones((self.K, self.K))
        np.fill_diagonal(mat, 20)
        self.Q = np.apply_along_axis(lambda row: np.random.dirichlet(row), axis=1, arr=mat)
        self.M = sample_normal(np.ones(self.K)*self.ξ, np.ones(self.K)*(1/self.κ)**(0.5))
        self.β = sample_gamma(self.g, np.ones(self.K)*self.h)
        self.Σ = sample_invgamma(self.α, 1/self.β)**(1/2)
        
        self.Σ = self.Σ[self.M.argsort()]
        self.ν = self.ν[self.M.argsort()]
        self.Q = self.Q[self.M.argsort()].T[self.M.argsort()].T
        self.M = self.M[self.M.argsort()]
        self.x_is = self.decode(self.ν, self.Q, self.true_z_is, self.M, self.Σ)[1]
        #self.x_is = self.decode(self.ν, self.Q, self.true_z_is, self.M, self.Σ)[1]
        
    def update_M(self):
        """
        Update parameter M in the LMA-HMM with Gibbs sampling using the current values of true_y_is, x_is, κ, ξ, and Σ.
    
        """
        n = np.bincount(list(map(int, self.x_is)), minlength=self.K)
        #S = np.bincount(list(map(int, self.x_is)), weights=self.true_y_is, minlength=self.K)
        S = np.bincount(list(map(int, self.x_is)), weights=self.true_z_is, minlength=self.K)
        self.M = sample_normal((S+self.κ*self.ξ*(self.Σ**2))/(n+self.κ*(self.Σ**2)), ((self.Σ**2)/(n+self.κ*(self.Σ**2)))**(1/2))
        
    def update_Σ(self):   
        """
        Update parameter Σ in the LMA-HMM with Gibbs sampling using the current values of true_y_is, x_is, α, β, M, and Σ.
        
        """
        
        x_counts = np.array([len(self.x_is[self.x_is == i]) for i in range(self.K)])
        #y_squared_sum = np.array([np.sum((self.true_y_is[self.x_is == k] - self.M[k])**2) for k in range(self.K)])
        y_squared_sum = np.array([np.sum((self.true_z_is[self.x_is == k] - self.M[k])**2) for k in range(self.K)])
        self.Σ = np.array(sample_invgamma2(self.α + x_counts/2, 1/(self.β + 1/2 * y_squared_sum)))**(1/2)
        self.β = np.array(sample_gamma(self.g + self.α, self.h + self.Σ**(-2)))

    def update_Q(self):
        """
        Update parameter Q in the LMA-HMM with Gibbs sampling using the current values of x_is and T_is.
        
        """
        
        n = np.array([[np.sum((self.x_is[:-1] == i) & (self.x_is[1:] == j)) for j in range(self.K)] for i in range(self.K)])        
        self.Q = np.array([sample_dirichlet(n[i] + 1).tolist() for i in range(self.K)])
        
    def update_ν(self):
        """
        Update parameter ν in the LMA-HMM with Gibbs sampling using the current value of x_is.
        
        """
        
        self.ν = np.array(sample_dirichlet(np.bincount(list(map(int, self.x_is)), minlength=self.K) + 1))
    
    def backward_procedure(self, Q, y, G_mat):
        """
        Perform the backward-procedure in the HMM.
    
        Parameters
        ----------
        Q : numpy.ndarray
            Transition matrix.
        G : function
            Function for emission probabilities.
        y : numpy.ndarray
           The time series of observations.
    
        Returns
        -------
        numpy.ndarray
            Matrix containing backward probabilities.
    
        Notes
        -----
        This method applies the backward procedure in the HMM, calculating backward probabilities for each state at each time step.
    
        """
        T = len(y)
        b = np.ones((T, self.K))
        for t in range(T-2,-1,-1):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            b[t,:] = np.dot(Q, G_mat[t+1]*b[t+1])
            b[t,:] = np.where(b[t,:].sum() > 0, b[t,:]/b[t,:].sum(), np.ones(self.K)/self.K) 
        return b
    
    def backward_procedure_ma(self, Q, y, G_mat):
        T = len(y)
        b = np.ones((T, self.K))
        for t in range(T-2,-1,-1):
            b[t] = (Q @ (b[t+1] * np.array([np.sum([G_mat[j][t+1] * np.count_nonzero(k == i) for j, k in enumerate(self.combi) if i in k]) for i in range(self.K)])))
            b[t] = np.where(b[t,:].sum() > 0, b[t,:]/b[t,:].sum(), np.ones(self.K)/self.K)
        return b
    
    def compute_G_mat(self, y, M, Σ):
        return np.array([pdf_normal(y, M[k], Σ[k]) for k in range(self.K)]).T
    
    def compute_G_mat_ma_permu(self, y, M, Σ):
        G_mat = self.compute_G_mat_ma_combi(self, y, M, Σ)
        return G_mat[np.where(np.all(np.sort(self.permu)[:, np.newaxis, :] == self.combi, axis=-1))[1]].T
    
    def compute_G_mat_ma_combi(self, y, M, Σ):
        M_ma = np.mean(M[self.combi], axis=1)
        Σ_ma = np.mean(Σ[self.combi], axis=1)/self.combi.shape[1]
        return np.array([pdf_normal(y, M_ma[k], Σ_ma[k]) for k in range(len(self.combi))])    
    
    def decode(self, ν, Q, y, M, Σ):
        """
        Computes the most likely sequence of hidden states given the current values of ν, Q, G, and y.

        Parameters
        ----------
        ν : ndarray, shape (K,)
            The initial distribution of the hidden states.
        Q : ndarray, shape (K,K)
            The transition matrix of the hidden states.
        G : function
            The emission function that takes a state i and an observation y and returns the probability of observing y in state i.
        y  : ndarray
           The time series of observations.
    
        Returns
        -------
        path_p : ndarray
            The estimated sequence of states probabilities.
        path_x : ndarray
            The estimated sequence of hidden states.
    
        """
        T = len(y)
        G_mat = self.compute_G_mat(y, M, Σ)

        b = self.backward_procedure(Q, y, G_mat)
        path_p, path_x = np.zeros((T, self.K)), np.zeros(T)
    
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        path_p[0] = ν * G_mat[0] * b[0]
        path_p[0] = np.where(path_p[0].sum() > 0, path_p[0]/path_p[0].sum(), np.ones(self.K)/self.K)
        path_x[0] = np.argmax(np.random.multinomial(1, path_p[0]))
    
        for t in range(1, T):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            path_p[t] = Q[int(path_x[(t-1)])] * G_mat[t] * b[t]
            path_p[t] = np.where(path_p[t].sum() > 0, path_p[t]/path_p[t].sum(), np.ones(self.K)/self.K)
            path_x[t] = np.argmax(np.random.multinomial(1, path_p[t]))
        return(path_p, path_x)
    
    def transition_probability(self, Q, chemin):
        p = 1.0
        for i in range(len(chemin) - 1):
            p *= Q[chemin[i], chemin[i + 1]]
        return p * Q[chemin[-1]]

    def Q_ma(self, Q):
        return np.array([self.transition_probability(Q, p) for p in self.permu])
    
    def decode_ma(self, ν, Q, y, M, Σ):
        G_mat_ma_combi = self.compute_G_mat_ma_combi(y, M, Σ)   

        b = self.backward_procedure_ma(Q, y, G_mat_ma_combi)
        T = len(y)
        path_p, path_x = np.zeros((T, self.K)), np.zeros(T)
        
        path_p[0] = ν * [pdf_normal(y[0], M[k], Σ[k]) for k in range(self.K)] * b[0]
        path_p[0] = np.nan_to_num(path_p[0]/path_p[0].sum(), nan=np.ones(len(path_p[0]))/len(path_p[0]))
        path_x[0] = np.argmax(np.random.multinomial(1, path_p[0]))

        for t in range(self.L+1,T):
            L = min(t-1, self.L)

            path_p[t] = self.transition_probability(Q, path_x[t-self.L-1:t].astype(int)) * np.array([G_mat_ma_combi[:,t][np.all(np.sort(np.append(path_x[t-L:t].astype(int), k)) == self.combi, axis=1)][0] for k in range(self.K)]) * b[t]
            path_p[t] = np.nan_to_num(path_p[t]/path_p[t].sum(), nan=np.ones(len(path_p[t]))/len(path_p[t]))
            path_x[t] = np.argmax(np.random.multinomial(1, path_p[t]))
        return(path_p, path_x)
    
    def decode_oos(self, Q, y, M, Σ):
        G_mat = self.compute_G_mat(y, M, Σ)
        x=self.x_is.astype(int)
        for t in range(len(y)):
            tmp = G_mat[t]*Q[x[-1]]
            tmp = np.where(tmp.sum() > 0, tmp/tmp.sum(), np.ones(self.K)/self.K)
            x = np.concatenate((x, [int(np.argmax(np.random.multinomial(1, tmp)))]))   
        return(x)
    
    def decode_oos_ma(self, Q, y, M, Σ, x) :
        """
        Computes the most likely sequence of out-of-sample hidden states given the current values of ν, Q, G, y and the previous hidden states with a depth of n.
    
        Parameters
        ----------
        ν : ndarray, shape (K,)
            The initial distribution of the hidden states.
        Q : ndarray, shape (K,K)
            The transition matrix of the hidden states.
        G : function
            The emission function that takes a state i and an observation y and returns the probability of observing y in state i, 
            given the previous hidden states with a depth of n.
    
        Returns
        -------
        x : ndarray
            The estimated sequence of hidden states.
            
        """
        for t in range(len(y)):
            L = 0
            combi = x[-L-1:]
            combi_G = np.column_stack((np.tile(combi[:L], (self.K, 1)), np.arange(self.K)))
            tmp = np.array([pdf_normal(y[t], np.mean(M[c]), np.mean(Σ[c])) for c in combi_G]) * self.transition_probability(Q, combi.astype(int))
            tmp = np.where(tmp.sum() > 0, tmp/tmp.sum(), np.ones(self.K)/self.K)
            tmp = tmp/tmp.sum()
            x = np.concatenate((x, [int(np.argmax(np.random.multinomial(1, tmp)))])) 
        return x
   
    def algorithm(self, n_iter, burn_in, compute_all_oos=True):
        """
        Runs the Gibbs sampler algorithm.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The size burn-in.
        est_ma : bool, optional
            A boolean indicating whether to estimate the parameters on the smoothed observations or just the decodin
            The default is True.
        compute_all_oos : bool, optional
            A boolean indicating whether to compute out-of-sample values of x for all iterations or not. The default is True.
            
        """
        self.res = {}
        
        for i in tq.tqdm(range(n_iter)): 
            
            self.update_M()
            self.M = np.sort(self.M)            
            
            self.update_Σ()
                
            _, self.x_is = self.decode(self.ν, self.Q, self.true_y_is, self.M, self.Σ)
            self.update_ν()
            self.update_Q()
    
                                      
            if i>burn_in:
                if compute_all_oos:
                    self.x_oos = self.decode_oos(self.Q, self.true_y_oos, self.M, self.Σ)[self.T_is:]
                    self.res[i] = {'Q': self.Q,
                                       'ν': self.ν,
                                       'M': self.M,
                                       'Σ': self.Σ,
                                       'x_is': self.x_is,
                                       'x_oos': self.x_oos}
                else:
                    self.res[i] = {'Q': self.Q,
                                   'ν': self.ν,
                                   'M': self.M,
                                   'Σ': self.Σ,
                                   'x_is': self.x_is}
    
    def algorithm_est_ma(self, n_iter, burn_in):
        """
        Runs the Gibbs sampler algorithm.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The size burn-in.
        est_ma : bool, optional
            A boolean indicating whether to estimate the parameters on the smoothed observations or just the decoding. The default is True.
        compute_all_oos : bool, optional
            A boolean indicating whether to compute out-of-sample values of x for all iterations or not. The default is True.
            
        """
        self.res = {}
        
        for i in tq.tqdm(range(n_iter)): 
            self.update_M()
            self.M = np.sort(self.M)            
            
            self.update_Σ()
                
            _, self.x_is = self.decode_ma(self.ν, self.Q, self.true_z_is, self.M, self.Σ)
            #_, self.x_is = self.decode(self.ν, self.Q, self.true_z_is, self.M, self.Σ)
            self.update_ν()
            self.update_Q()

                                      
            if i>burn_in:
                self.x_oos = self.decode_oos_ma(self.Q, self.true_z_oos, self.M, self.Σ, self.x_is.astype(int))[self.T_is-self.L:]
                self.res[i] = {'Q': self.Q,
                               'ν': self.ν,
                               'M': self.M,
                               'Σ': self.Σ,
                               'x_is': self.x_is,
                               'x_oos': self.x_oos}
        
    def algorithm_flow(self, n_iter, burn_in, x_is):
        """
        Runs the Gibbs sampler algorithm.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The size burn-in.
        est_ma : bool, optional
            A boolean indicating whether to estimate the parameters on the smoothed observations or just the decoding. The default is True.
        compute_all_oos : bool, optional
            A boolean indicating whether to compute out-of-sample values of x for all iterations or not. The default is True.
            
        """
        self.res = {}

        for i in tq.tqdm(range(n_iter)): 
            self.update_M()
            self.M = np.sort(self.M)            

            self.update_Σ()
        
            _, self.x_is = self.decode_ma(self.ν, self.Q, self.true_z_is, self.M, self.Σ)
            self.update_ν()
            self.update_Q()

            if i>burn_in:
                self.x_oos = self.decode_oos_ma(self.Q, self.true_z_oos, self.M, self.Σ, x_is.astype(int))[len(x_is):]
                self.res[i] = {'Q': self.Q,
                               'ν': self.ν,
                               'M': self.M,
                               'Σ': self.Σ,
                               'x_is': self.x_is,
                               'x_oos': self.x_oos}
                
    
    
 