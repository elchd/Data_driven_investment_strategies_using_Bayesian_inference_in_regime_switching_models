import numpy as np
from .random_sample import *
from .ma_hmm import *
import tqdm.notebook as tq
import warnings

class gibbs(ma_hmm):
    def __init__(self, K, L, train_test_split = 0.7, y=None, index_is_oos=None):
        """
        Initialize the Gibbs sampler class for MA(L)-HMM with the given parameters.
        
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
        This class extends the ma_hmm class for Gibbs sampling specific functionality. 
        It initializes with parameters for the number of states (K), moving average depth (L), 
        training-test split ratio, input data (y), and the index separating in-sample and out-of-sample data (if applicable).

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
        Initialize the hyperparameters (ξ, κ, α, g and h) based on properties of the in-sample data and the priors 
        for Gibbs sampling algorithm.
    
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
        Perform pretraining using pre-specified parameters, simulate hidden states and observations and 
        initialize the parameters for Gibbs sampling

        Parameters
        ----------
        ν : numpy.ndarray, shape (K,)
            The initial distribution vector of the hidden states.
        Q : numpy.ndarray, shape (K,K)
            The transition matrix of the hidden states.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
        T : int
            The length of the time series to be simulated.
            
        """
        super().__init__(ν, Q, M, Σ, T, self.L, self.tts)
        super().simul()
        self.initialize() 
               
    def compute_priors(self):
        """
        Compute the prior parameters of the MA(L)-HMM:
            - Dirichlet prior for ν : initial distribution vector of the hidden states.
            - Dirichlet prior for Q : transition matrix of the hidden states.
            - Normal prior for M : means vector of the emission distribution for each hidden state.
            - Gamma prior for β : shape parameters for the inverse-Gamma distribution for Σ.
            - Inverse-Gamma prior for Σ : standard deviations vector of the emission distribution for each hidden state.

        Notes
        ----------
        The resulting parameters are sorted according to the means M to maintain consistency.
        Additionally, the state sequences are decoded using the computed priors.
            
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
        
    def update_M(self):
        """
        Update parameter M in the MA(L)-HMM with Gibbs sampling using the current values of true_y_is, x_is and Σ
        and hyperparameters κ and ξ.
    
        """
        n = np.bincount(list(map(int, self.x_is)), minlength=self.K)
        S = np.bincount(list(map(int, self.x_is)), weights=self.true_z_is, minlength=self.K)
        self.M = sample_normal((S+self.κ*self.ξ*(self.Σ**2))/(n+self.κ*(self.Σ**2)), ((self.Σ**2)/(n+self.κ*(self.Σ**2)))**(1/2))
        
    def update_Σ(self):   
        """
        Update parameter Σ in the MA(L)-HMM with Gibbs sampling using the current values of true_y_is, x_is, β, M, and Σ
        and hyperparameter α.
        
        """
        x_counts = np.array([len(self.x_is[self.x_is == i]) for i in range(self.K)])
        y_squared_sum = np.array([np.sum((self.true_z_is[self.x_is == k] - self.M[k])**2) for k in range(self.K)])
        self.Σ = np.array(sample_invgamma2(self.α + x_counts/2, 1/(self.β + 1/2 * y_squared_sum)))**(1/2)
        self.β = np.array(sample_gamma(self.g + self.α, self.h + self.Σ**(-2)))

    def update_Q(self):
        """
        Update parameter Q in the MA(L)-HMM with Gibbs sampling using the current values of x_is.
        
        """
        n = np.array([[np.sum((self.x_is[:-1] == i) & (self.x_is[1:] == j)) for j in range(self.K)] for i in range(self.K)])        
        self.Q = np.array([sample_dirichlet(n[i] + 1).tolist() for i in range(self.K)])
        
    def update_ν(self):
        """
        Update parameter ν in the MA(L)-HMM with Gibbs sampling using the current value of x_is.
        
        """
        self.ν = np.array(sample_dirichlet(np.bincount(list(map(int, self.x_is)), minlength=self.K) + 1))
    
    def backward_procedure(self, Q, y, G_mat):
        """
        Perform the backward-procedure in the HMM.
    
        Parameters
        ----------
        Q : numpy.ndarray
            The transition matrix of the hidden states.
        y : numpy.ndarray
           The time series of observations.
        G_mat : numpy.ndarray
            The emission matrix.
    
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
        """
        Perform the backward-procedure in the MA(L)-HMM.
    
        Parameters
        ----------
        Q : numpy.ndarray
            The transition matrix of the hidden states.
        y : numpy.ndarray
           The time series of observations.
        G_mat : function
            The emission matrix.
    
        Returns
        -------
        numpy.ndarray
            Matrix containing backward probabilities.
    
        Notes
        -----
        This method applies the backward procedure in the MA(L)-HMM, calculating backward probabilities for each state at 
        each time step.
        The procedure is modified to accommodate the moving average structure of the MA(L)-HMM, where the emission 
        probabilities depend on a combination of multiple latent states.
    
        """
        T = len(y)
        b = np.ones((T, self.K))
        for t in range(T-2,-1,-1):
            b[t] = (Q @ (b[t+1] * np.array([np.sum([G_mat[j][t+1] * np.count_nonzero(k == i) for j, k in enumerate(self.combi) if i in k]) for i in range(self.K)])))
            b[t] = np.where(b[t,:].sum() > 0, b[t,:]/b[t,:].sum(), np.ones(self.K)/self.K)
        return b
    
    def compute_G_mat(self, y, M, Σ):
        """
        Compute the emission matrix containing the emission probabilities for each observation t (0 < t < T) and states (k ∈ {X}) 
        given the means and standard deviations vectors.

        Parameters
        ----------
        y : numpy.ndarray, shape (T,)
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.

        Returns
        -------            
        numpy.ndarray, shape (T, K)
            Emission matrix.
        
        """
        return np.array([pdf_normal(y, M[k], Σ[k]) for k in range(self.K)]).T
    
    def compute_G_mat_ma_permu(self, y, M, Σ):
        """
        Compute the emission matrix containing the emission probabilities for each observation t (0 < t < T) and permutations
        of states given the means and standard deviations vectors.

        Parameters
        ----------
        y : numpy.ndarray, shape (T,)
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.

        Returns
        -------            
        numpy.ndarray, shape (T, K^L)
            Emission matrix of permutations of states.
        
        """
        G_mat = self.compute_G_mat_ma_combi(self, y, M, Σ)
        return G_mat[np.where(np.all(np.sort(self.permu)[:, np.newaxis, :] == self.combi, axis=-1))[1]].T
    
    def compute_G_mat_ma_combi(self, y, M, Σ):
        """
        Compute the emission matrix containing the emission probabilities for each observation t (0 < t < T) and combinations
        of states given the means and standard deviations vectors.

        Parameters
        ----------
        y : numpy.ndarray, shape (T,)
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.

        Returns
        -------            
        numpy.ndarray, shape (T,C(L+K-1, K-1))
            Emission matrix of combinations of states.
        
        """
        M_ma = np.mean(M[self.combi], axis=1)
        Σ_ma = np.mean(Σ[self.combi], axis=1)/self.combi.shape[1]
        return np.array([pdf_normal(y, M_ma[k], Σ_ma[k]) for k in range(len(self.combi))])    
    
    def decode(self, ν, Q, y, M, Σ):
        """
        Compute the most likely sequence of hidden states of the HMM given the current values of ν, Q, G, and y.

        Parameters
        ----------
        ν : numpy.ndarray, shape (K,)
            The initial distribution vector of the hidden states.
        Q : numpy.ndarray, shape (K,K)
            The transition matrix of the hidden states.
        y  : numpy.ndarray
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
    
        Returns
        -------
        path_p : numpy.ndarray
            The estimated sequence of states probabilities.
        path_x : numpy.ndarray
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
    
    def transition_probability(self, Q, path):
        """
        Compute the transition probability of a given hidden state path.

        Parameters
        ----------
        Q : numpy.ndarray
            The transition matrix of the hidden states.
        path : numpy.ndarray
            A sequence of hidden states.
    
        Returns
        -------
        float
            The estimated probability of the given state path.
    
        """
        p = 1.0
        for i in range(len(path) - 1):
            p *= Q[path[i], path[i + 1]]
        return p * Q[path[-1]]
    
    def decode_ma(self, ν, Q, y, M, Σ):
        """
        Compute the most likely sequence of hidden states of the MA(L)-HMM given the current values of ν, Q, G, and y.

        Parameters
        ----------
        ν : numpy.ndarray, shape (K,)
            The initial distribution vector of the hidden states.
        Q : numpy.ndarray, shape (K,K)
            The transition matrix of the hidden states.
        y  : numpy.ndarray
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
    
        Returns
        -------
        path_p : numpy.ndarray
            The estimated sequence of states probabilities.
        path_x : numpy.ndarray
            The estimated sequence of hidden states.
    
        """
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
        """
        Compute the most likely sequence of out-of-sample hidden states of the HMM given the current values of ν, Q, G, y 
        and the previous hidden state.
    
        Parameters
        ----------
        Q : numpy.ndarray, shape (K,K)
            The transition matrix of the hidden states.
        y  : numpy.ndarray
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
    
        Returns
        -------
        x : numpy.ndarray
            The estimated sequence of hidden states.
            
        """
        G_mat = self.compute_G_mat(y, M, Σ)
        x=self.x_is.astype(int)
        for t in range(len(y)):
            tmp = G_mat[t]*Q[x[-1]]
            tmp = np.where(tmp.sum() > 0, tmp/tmp.sum(), np.ones(self.K)/self.K)
            x = np.concatenate((x, [int(np.argmax(np.random.multinomial(1, tmp)))]))   
        return(x)
    
    def decode_oos_ma(self, Q, y, M, Σ, x) :
        """
        Compute the most likely sequence of out-of-sample hidden states of the MA(L)-HMM given the current values 
        of ν, Q, G, y and the previous hidden states with a depth of L.
    
        Parameters
        ----------
        Q : numpy.ndarray, shape (K,K)
            The transition matrix of the hidden states.
        y  : numpy.ndarray
           The time series of observations.
        M : numpy.ndarray, shape (K,)
            The means vector of the emission distribution for each hidden state.    
        Σ : numpy.ndarray, shape (K,)
            The standard deviations vector of the emission distribution for each hidden state.
        x : numpy.ndarray
           The time series of hidden states.
    
        Returns
        -------
        x : numpy.ndarray
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
        Run the Gibbs sampler algorithm for the HMM.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The burn-in size.
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
        Run the Gibbs sampler algorithm for the MA(L)-HMM.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The burn-in size.

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
                self.x_oos = self.decode_oos_ma(self.Q, self.true_z_oos, self.M, self.Σ, self.x_is.astype(int))[self.T_is-self.L:]
                self.res[i] = {'Q': self.Q,
                               'ν': self.ν,
                               'M': self.M,
                               'Σ': self.Σ,
                               'x_is': self.x_is,
                               'x_oos': self.x_oos}
        
    def algorithm_ma_flow(self, n_iter, burn_in, x_is):
        """
        Run the Gibbs sampler algorithm for the MA(L)-HMM given the previous sequence of decoded hidden states x_is
        to ensure continuity in the decoding of hidden states for the time series.
        
        Parameters
        ----------
        n_iter : int
            The number of iterations of the Gibbs sampler.
        burn_in : int
            The burn-in size.
        x_is : numpy.ndarray
           The time series of previously decoded hidden states.

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
                
    
    
 