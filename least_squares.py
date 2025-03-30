"""
Implementation of the Least Squares localization algorithm.

This implementation is based on the mathematical formulation described in:

    Ye, Z., Xu, Y., Lin, J., Li, G., Geng, E., & Pang, Y. (2018). 
    "An Improved Bluetooth Indoor Positioning Algorithm Based on RSSI and PSO-BPNN." 
    Sensors, 18(9), 2820. https://doi.org/10.3390/s18092820

This is not a direct copy but an implementation derived from the concepts presented in the publication.
"""

from typing import List
import numpy as np

from common import LocalizationAlgorithm2D, Vector2D


def get_2d_position_using_least_squares_algorithm(x_vals: List[float], 
                                                  y_vals: List[float], 
                                                  d_vals: List[float]) -> Vector2D:
    """Compute 2D position using the least squares method given beacon coordinates and distances."""
    a = _solve_a(x_vals, y_vals)
    b = _solve_b(x_vals, y_vals, d_vals)
    predicted_pos = _multiply(_inverse(_multiply(_transpose(a), a)), _multiply(_transpose(a), b))
    pos_x, pos_y = predicted_pos[0][0], predicted_pos[1][0]
    return Vector2D(pos_x, pos_y)


def _solve_a(x: List[float], y: List[float]) -> np.array:
    """Construct matrix A for the least squares equation."""
    m = len(x)
    a = np.zeros((m - 1, 2))
    for i in range(m - 1):
        a[i][0] = 2 * (x[i] - x[-1])
        a[i][1] = 2 * (y[i] - y[-1])
    return a


def _func_b(x1: float, xm: float, y1: float, ym: float, d1: float, dm: float) -> float:
    """Compute individual elements of matrix B using the given distance and coordinate values."""
    return x1 * x1 - xm * xm + y1 * y1 - ym * ym + d1 * d1 - dm * dm


def _solve_b(x: List[float], y: List[float], d: List[float]) -> np.array:
    """Construct matrix B for the least squares equation."""
    m = len(x)
    b = np.zeros((m - 1, 1))
    for i in range(m - 1):
        b[i][0] = _func_b(x[i], x[-1], y[i], y[-1], d[-1], d[i])
    return b


def _multiply(matrix_a: np.array, matrix_b: np.array) -> np.array:
    """Perform matrix multiplication."""
    return np.matmul(matrix_a, matrix_b)


def _transpose(matrix: np.array) -> np.array:
    """Return the transpose of a matrix."""
    return matrix.transpose()


def _inverse(matrix: np.array) -> np.array:
    """Return the inverse of a matrix."""
    return np.linalg.inv(matrix)


class LeastSquaresAlgorithm2D(LocalizationAlgorithm2D):
    def predict(self, x_list: List[float], y_list: List[float], d_list: List[float]) -> Vector2D:
        """Predict the 2D position based on given beacon coordinates and distances."""
        return get_2d_position_using_least_squares_algorithm(x_list, y_list, d_list)
