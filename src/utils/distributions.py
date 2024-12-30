import numpy as np

from scipy.stats import ecdf

def empirical_cdf_from_samples(samples: np.ndarray) -> callable:
    result = ecdf(samples)
    
    return lambda x: result.cdf.evaluate(x)
  
