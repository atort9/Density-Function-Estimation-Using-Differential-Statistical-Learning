import numpy as np
from abc import abstractmethod
from src.utils.integration import *
from src.utils.distributions import empirical_cdf_from_samples
import matplotlib.pyplot as plt


class UnidimensionalSeries:

    @abstractmethod
    def base_function(self, index: int):
        pass
    
    @abstractmethod
    def base_function_derivative(self, index: int):
        pass
    
    @abstractmethod
    def base_function_integral(self, index: int):
        pass
    
    def transform_index(self, index: int) -> int:
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass
    
    def base_function_product(self, i: int, j: int, number_points: int = 100) -> complex:
        x = np.linspace(self.domain[0], self.domain[1], number_points)
        y_i = self.base_function(i)(x)
        y_j = self.base_function(j)(x)
        y = y_i * y_j
        
        return integrate_function(x, y)

    def base_function_derivative_product(self, i: int, j: int, number_points: int = 100) -> complex:
        x = np.linspace(self.domain[0], self.domain[1], number_points)
        y_i = self.base_function_derivative(i)(x)
        y_j = self.base_function_derivative(j)(x)
        y = y_i * y_j
    
        return integrate_function(x, y)

    def build_kernel_matrix_base_function(self, number_points: int = 100) -> np.ndarray:
        
        kernel_matrix = np.zeros((self.size(), self.size()), dtype=complex)
        
        for i in range(self.size()):
            for j in range(self.size()):
                kernel_matrix[i, j] = self.base_function_product(i, j, number_points)
        return kernel_matrix
    
    def build_kernel_matrix_base_function_derivative(self, number_points: int = 100) -> np.ndarray:
        kernel_matrix = np.zeros((self.size(), self.size()), dtype=complex)
        
        for i in range(self.size()):
            for j in range(self.size()):
                kernel_matrix[i, j] = self.base_function_derivative_product(i, j, number_points)

        return kernel_matrix

    def build_kernel_vector_from_samples_pdf_dotprod_bf(self, samples: np.array) -> np.ndarray:
        kernel_vector = np.zeros(self.size(), dtype=complex)
        
        for i in range(self.size()):
            kernel_vector[i] = np.mean(self.base_function(i)(samples))

        return kernel_vector
    
    def build_kernel_vector_from_samples_pdf_dotprod_bf_derivative(self, samples: np.array) -> np.ndarray:
        kernel_vector = np.zeros(self.size(), dtype=complex)
        
        for i in range(self.size()):
            kernel_vector[i] = np.mean(self.base_function_derivative(i)(samples))

        return kernel_vector

    def build_kernel_vector_from_samples_cdf_dotprod_bf(self, samples: np.array, number_points: int = 100) -> np.ndarray:
        empirical_cdf = empirical_cdf_from_samples(samples)
        domain = np.linspace(self.domain[0], self.domain[1], number_points)
        kernel_vector = np.zeros(self.size(), dtype=complex)
        
        for i in range(self.size()):
            kernel_vector[i] = integrate_product_function(domain, empirical_cdf(domain), self.base_function(i)(domain))
        return kernel_vector
      
    def build_kernel_vector_from_samples_cdf_dotprod_bf_derivative(self, samples: np.array) -> np.ndarray:
        empirical_cdf = empirical_cdf_from_samples(samples)
        domain = np.linspace(self.domain[0], self.domain[1], number_points)
        kernel_vector = np.zeros(self.size(), dtype=complex)
        
        for i in range(self.size()):
            kernel_vector[i] = integrate_product_function(domain, empirical_cdf(domain), self.base_function_derivative(i)(domain))

        return kernel_vector

    def fit_coeff_from_samples_R_PDF(self, samples: np.array) -> np.ndarray:
 
        kernel_matrix = self.build_kernel_matrix_base_function()
        kernel_vector = self.build_kernel_vector_from_samples_pdf_dotprod_bf(samples)
        try:
            self.coefficients = np.linalg.solve(kernel_matrix, kernel_vector)
        except:
            self.coefficients = np.linalg.lstsq(kernel_matrix, kernel_vector, rcond=None)

    
    def fit_coeff_from_samples_R_CDF(self, samples: np.array) -> np.ndarray:
  
        kernel_matrix = self.build_kernel_matrix_base_function()
        kernel_vector = self.build_kernel_vector_from_samples_cdf_dotprod_bf(samples)
        try:
            self.coefficients = np.linalg.solve(kernel_matrix, kernel_vector)
        except:
            self.coefficients = np.linalg.lstsq(kernel_matrix, kernel_vector, rcond=None)
        
    def fit_coeff_from_samples_R_CDF_PDF(self, samples: np.array, weight_cdf: float = 1., weight_pdf: float = 1.) -> np.ndarray:
   
        kernel_matrix = self.build_kernel_matrix_base_function()
        kernel_matrix_derivative = self.build_kernel_matrix_base_function_derivative()
        kernel_vector = self.build_kernel_vector_from_samples_cdf_dotprod_bf(samples)
        kernel_vector_derivative = self.build_kernel_vector_from_samples_pdf_dotprod_bf_derivative(samples)
        
        complete_kernel_matrix = weight_cdf * kernel_matrix + weight_pdf * kernel_matrix_derivative
        complete_kernel_vector = weight_cdf * kernel_vector + weight_pdf * kernel_vector_derivative

        try:
            self.coefficients = np.linalg.solve(complete_kernel_matrix, complete_kernel_vector)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(complete_kernel_matrix, complete_kernel_vector, rcond=None)


    def __call__(self, x: np.array) -> np.ndarray:

        result = np.zeros_like(x)
        
        try:
          for i in range(self.size()):
              result = result + self.coefficients[i] * self.base_function(i)(x)
        except:
          for i in range(self.size()):
              result = result + self.coefficients[0][i] * self.base_function(i)(x)

        return result.real
    
    def call_using_base_function_derivative(self, x: np.array) -> np.ndarray:

        result = np.zeros_like(x)
        
        for i in range(self.size()):
            result = result + self.coefficients[i] * self.base_function_derivative(i)(x)

        return result.real
    
    def call_using_base_function_integral(self, x: np.array) -> np.ndarray:

        result = np.zeros_like(x)
        
        for i in range(self.size()):
            result = result + self.coefficients[i] * self.base_function_integral(i)(x)
        
        constant_integration = -result[0] 
        result = result + constant_integration
        
        return result.real

