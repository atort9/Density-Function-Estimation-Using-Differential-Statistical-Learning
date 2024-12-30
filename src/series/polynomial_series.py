import numpy as np
import matplotlib.pyplot as plt
from src.series.unidimensional_series import UnidimensionalSeries

class PolynomialSeries(UnidimensionalSeries):

    def __init__(self, fundamental_frequency: float, number_coefficients: int, domain: tuple, coefficients: np.array = None):

        self.number_coefficients = number_coefficients
        if coefficients is None:
            self.coefficients = np.zeros(2 * self.number_coefficients + 1)
        self.domain = domain
        self.fundamental_frequency = fundamental_frequency

    def base_function(self, index: int) -> callable:
        return lambda x: x**index
    
    def base_function_derivative(self, index: int) -> callable:
        if index == 0:
            return lambda x: np.zeros_like(x)  
        else:
            return lambda x: index * x**(index - 1)
        
    def base_function_integral(self, index: int) -> callable:
        if index == -1:
            return lambda x: np.zeros_like(x)
        else:
            return lambda x: (x**(index + 1)) / (index + 1)

    def transform_index(self, index: int) -> int:
        return index
    
    def size(self) -> int:
        return 2 * self.number_coefficients + 1
