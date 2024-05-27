import numpy as np
import math as ma
from numba import njit

#%%
@njit
def pdf_normal(x, μ, σ):
    """
    Probability density function (PDF) of a normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF.
    μ : float
        Mean of the normal distribution.
    σ : float
        Standard deviation of the normal distribution.
    log : bool, optional
        If True, return the log PDF. If False, return the actual PDF. Default is True.

    Returns
    -------
    float
        Value of the PDF at x.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    return (1/np.sqrt(2*np.pi*σ**2))*np.exp(-(x-μ)**2/(2*σ**2))

@njit       
def pdf_normal_log(x, μ, σ):
    """
    Probability density function (PDF) of a normal distribution.

    Parameters
    ----------
    x : float
        Value at which to evaluate the PDF.
    μ : float
        Mean of the normal distribution.
    σ : float
        Standard deviation of the normal distribution.
    log : bool, optional
        If True, return the log PDF. If False, return the actual PDF. Default is True.

    Returns
    -------
    float
        Value of the PDF at x.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    return -(x-μ)**2/(2*σ**2) - np.log(σ*np.sqrt(2*np.pi))


def get_pdf(params, name, log = False):
    """
    Generate a probability density function (PDF) based on the specified distribution type.

    Parameters
    ----------
    params : list or tuple
        Parameters specific to the chosen distribution.
    name : str
        Name of the distribution ('gaussian', 'multivariate normal', 'gamma', 'dirichlet').
    log : bool, optional
        If True, return the log PDF. If False, return the actual PDF. Default is False.

    Returns
    -------
    function
        The PDF function that takes a value x as input.

    Notes
    -----
    - For Gaussian distribution, params should be [μ, σ].
    - For multivariate normal distribution, params should be [M, Σ].
    - For gamma distribution, params should be [α, β].
    - For dirichlet distribution, params should be a list of α values.

    """
    if name == 'gaussian':
        def f_x(x):
            μ, σ = params[0], params[1]
            if σ < 0:
                return(0)
            else:
                if log:
                    return(-(x-μ)**2/(2*σ**2) - np.log(σ*np.sqrt(2*np.pi)))
                else:
                    return((1/np.sqrt(2*np.pi*σ**2))*np.exp(-(x-μ)**2/(2*σ**2))) 
    
    elif name == 'multivariate normal':
        def f_x(x):
            M, Σ = params[0], params[1]
            K = len(M)
            return(1/np.sqrt((2*np.pi)**K*np.linalg.det(np.diag(Σ))) * np.exp((x-M).T * Σ**(-1) * (x-M)))

    elif name == 'gamma':
        def f_x(x):
            α, β = params[0], params[1]
            if (α <= 0) | (β <= 0):
                return(0)
            else:
                if log:
                     return((α*np.log(β))-np.log(ma.gamma(α)) + (α-1)*np.log(x) - β*x)
                else:
                     return((β**α / ma.gamma()(α))*x**(α-1)*np.exp(-β*x))
       
    elif name == 'dirichlet':
        def f_x(x):
            α = np.array(params)            
            if(len(α[α==0]) > 0):
                return(0)
            else:
                K = len(α)
                if log:
                    return(- (np.sum(np.log([ma.gamma(α[i]) for i in range(K)])) - np.log(ma.gamma(np.sum(α)))) + np.sum([(α[i]-1) * np.log(x[i]) for i in range(K)]))
                else:
                    return((1 / (np.prod([ma.gamma(α[i]) for i in range(K)]) / ma.gamma(np.sum(α)))) * np.prod([x[i]**(α[i]-1) for i in range(K)]))     
    
    else:
        print(f'Error: distribution type <{name}> not handled')
        return 
    
    return f_x
    
def get_ll(params, name):
    """
    Generate a log-likelihood function based on the specified distribution type.

    Parameters
    ----------
    params : list or tuple
        Parameters specific to the chosen distribution.
    name : str
        Name of the distribution ('gaussian', 'gamma', 'dirichlet').

    Returns
    -------
    function
        The log-likelihood function that takes a dataset x as input.

    Notes
    -----
    - For Gaussian distribution, params should be [μ, σ].
    - For gamma distribution, params should be [α, β].
    - For dirichlet distribution, params should be a list of α values.

    """
   
    if name == 'gaussian':
        def ll_x(x):
            μ, σ = params[0], params[1]
            if params[1] <= 0:
                return(0)
            else:
                f_x = get_pdf(params, 'gaussian', log = True)
                return(np.sum(f_x(x)))
        
    elif name == 'gamma':
         def ll_x(x):
            α, β = params[0], params[1]
            if (α <= 0) | (β <= 0):
                return(0)
            else:
                f_x = get_pdf(params, 'gamma', log = True)
                return(np.sum(f_x(x)))
                              
    elif name == 'dirichlet':
        def ll_x(x):
            α = np.array(params)            
            if(len(α[α==0]) > 0):
                return(0)
            else:
                f_x = get_pdf(params, 'dirichlet', log = True)
                return(np.sum(f_x(x)))
            
    else:
        print(f'Error: distribution type <{name}> not handled')
        return 
    
    return(ll_x)

