import numpy as np
import matplotlib.pyplot as plt
from src.series.unidimensional_series import UnidimensionalSeries

class TrigonometricSeries(UnidimensionalSeries):

    def __init__(self, fundamental_frequency: float, number_coefficients: int, domain: tuple, coefficients: np.array = None):

        self.number_coefficients = number_coefficients
        if coefficients is None:
            self.coefficients = np.zeros(2 * self.number_coefficients + 1)
        self.domain = domain
        self.fundamental_frequency = fundamental_frequency
        
    def base_function(self, index: int) -> callable:
        return lambda x: np.exp(self.fundamental_frequency * 1j * self.transform_index(index) * x)
    
    def base_function_derivative(self, index: int) -> callable:
        return lambda x: self.fundamental_frequency * 1j * self.transform_index(index) * np.exp(self.fundamental_frequency * 1j * self.transform_index(index) * x)
    
    def base_function_integral(self, index: int) -> callable:
        transformed_index = self.transform_index(index)
        if transformed_index == 0:
            return lambda x: x
        else:
            return lambda x: np.exp(self.fundamental_frequency * 1j * transformed_index * x) / (self.fundamental_frequency * 1j * transformed_index)

    def transform_index(self, index: int) -> int:
        if index <= self.number_coefficients:
            return index
        else:
            return index - (2 * self.number_coefficients + 1)
    
    def size(self) -> int:
        return 2 * self.number_coefficients + 1


