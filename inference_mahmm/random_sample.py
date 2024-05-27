from numba import njit
import numpy as np

@njit
def sample_normal(μ, σ):
    """
    Sample from a multivariate normal distribution with means μ and standard deviations σ.

    Parameters
    ----------
    μ : ndarray
        Means of the normal distribution.
    σ : ndarray
        Standard deviations of the normal distribution.

    Returns
    -------
    ndarray
        Samples from the multivariate normal distribution.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    return np.array([np.random.normal(μ[i], σ[i]) for i in range(len(μ))])
    
def sample_dirichlet(α):
    """
    Sample from a Dirichlet distribution with concentrations α.

    Parameters
    ----------
    α : ndarray
        Vector of concentration parameters.

    Returns
    -------
    ndarray
        Sampled probabilities from the Dirichlet distribution.

    """
    return np.random.dirichlet(α)

@njit(parallel=True)
def sample_uniform(l, h):
    """
    Sample from a uniform distribution with lower bound l and upper bound h.

    Parameters
    ----------
    l : float
        Lower bound of the uniform distribution.
    h : float
        Upper bound of the uniform distribution.

    Returns
    -------
    float
        Random sample from the uniform distribution.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    if h < l:
        return 0
    else:
        return np.random.uniform(l,h)

@njit
def sample_gamma(α, β):
    """
    Sample from a gamma distribution with shape α and scale β.

    Parameters
    ----------
    α : float
        Shape parameter of the gamma distribution.
    β : ndarray
        Scale parameter(s) of the gamma distribution.

    Returns
    -------
    ndarray
        Sampled values from the gamma distribution.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation. 

    """
    return np.array([np.random.gamma(α, 1/β[i]) for i in range(len(β))])

@njit
def sample_invgamma(α, β):
    """
    Sample from an inverse gamma distribution with shape α and scale β.

    Parameters
    ----------
    α : float
        Shape parameter of the inverse gamma distribution.
    β : ndarray
        Scale parameter(s) of the inverse gamma distribution.

    Returns
    -------
    ndarray
        Sampled values from the inverse gamma distribution.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    return np.array([np.random.gamma(α, β[i])**(-1) for i in range(len(β))])

def sample_invgamma2(α, β):
    """
    Sample from an inverse gamma distribution with shape α and scale β.

    Parameters
    ----------
    α : float
        Shape parameter of the inverse gamma distribution.
    β : ndarray
        Scale parameter(s) of the inverse gamma distribution.

    Returns
    -------
    ndarray
        Sampled values from the inverse gamma distribution.

    Notes
    -----
    This function uses Just-In-Time (JIT) compilation.

    """
    return(np.array([np.random.gamma(α[i], β[i])**(-1) for i in range(len(β))]))
