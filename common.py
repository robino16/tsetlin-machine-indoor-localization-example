from typing import List
from abc import ABC, abstractmethod
from collections import namedtuple

# Define a simple 2D vector type using namedtuple for clarity and ease of use
Vector2D = namedtuple("Vector2D", ["x", "y"])


class LocalizationAlgorithm2D(ABC):
    @abstractmethod
    def predict(self, x_list: List[float], y_list: List[float], d_list: List[float]) -> Vector2D:
        """Abstract method to predict the 2D position given beacon coordinates and distances."""
        raise NotImplementedError("Subclasses must implement this method.")
