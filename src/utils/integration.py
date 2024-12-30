import numpy as np
import tensorflow as tf
from typing import Callable
from multimethod import multimethod

@multimethod
def integrate_function(domain: np.ndarray, function: Callable) -> float:
        y = function(domain)

        dx = np.diff(domain)
        integral = np.dot(dx, (y[:-1] + y[1:])/2)

        return integral

@multimethod
def integrate_function(domain: np.ndarray, y: np.ndarray) -> float:
        dx = np.diff(domain)
        integral = np.dot(dx, (y[:-1] + y[1:])/2)

        return integral
      
@multimethod
def integrate_function(domain: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        dx = domain[1:] - domain[:-1]  
        a = (y[:-1] + y[1:])/2
        integral = tf.reduce_sum(tf.squeeze(dx) * tf.squeeze(a))

        return integral


def integrate_product_function(domain: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:

        y = y1 * y2
        dx = np.diff(domain)
        integral = np.dot(dx, (y[:-1] + y[1:])/2)

        return integral
