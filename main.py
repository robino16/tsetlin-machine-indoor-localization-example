import logging

import numpy as np
import pandas as pd
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from tmu.models.regression.vanilla_regressor import TMRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from common import Vector2D
from least_squares import LeastSquaresAlgorithm2D

logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Suppress matplotlib debug logs
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)  # Suppress PIL debug logs

DATASET_FILE_PATH = "./data/iBeacon_RSSI_Labeled.csv"
EPOCHS = 30

# Hardcoded beacon positions based on the BLE RSSI dataset by mehdimka on Kaggle
BEACON_POSITIONS = {
    "b3001": Vector2D(5, 8), 
    "b3002": Vector2D(10, 3), 
    "b3003": Vector2D(14, 3), 
    "b3004": Vector2D(18, 3),
    "b3005": Vector2D(10, 6.5), 
    "b3006": Vector2D(14, 6.5),
    "b3007": Vector2D(18, 6.5), 
    "b3008": Vector2D(10, 9.5),
    "b3009": Vector2D(4, 15), 
    "b3010": Vector2D(10, 15), 
    "b3011": Vector2D(14, 15), 
    "b3012": Vector2D(18, 15), 
    "b3013": Vector2D(23, 15),
}


def calc_distance(pos1: Vector2D, pos2: Vector2D) -> float:
    """Compute Euclidean distance between two points."""
    return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5


def location_key_to_vector_2d(location_key: str) -> Vector2D:
    """Convert location key (e.g., 'A01') to 2D coordinates."""
    letter = location_key[0].upper()
    number = int(location_key[1:])
    x = ord(letter) - ord("A") + 0.5  # Convert letter to numeric index (A=0.5, B=1.5, ..., O=14.5)
    y = number - 0.5  # Convert number to zero-indexed coordinate
    return Vector2D(x, y)


def plot_rssi_vs_actual_distance(df):
    """Plot RSSI values (dBm) vs. actual distance (meters), ignoring out-of-range values."""
    plt.figure(figsize=(8, 6))

    actual_distances = []
    rssi_values = []

    for beacon, position in BEACON_POSITIONS.items():
        for _, row in df.iterrows():
            rssi = row[beacon]
            if rssi < -150:  # Skip out-of-range values
                continue
            actual_distances.append(calc_distance(location_key_to_vector_2d(row["location"]), position))
            rssi_values.append(rssi)

    plt.scatter(actual_distances, rssi_values, color="blue", alpha=0.6, s=10, zorder=2)
    plt.grid(zorder=1) 

    plt.xlabel("Actual Distance (m)")
    plt.ylabel("RSSI (dBm)")
    plt.title("RSSI vs Actual Distance")
    plt.show()


def plot_predicted_vs_actual_distance(y_test, y_pred):
    """Plot predicted distance vs. actual distance with a reference line."""
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, alpha=0.6, s=10, label="Predictions")
    plt.plot([0, max(y_test)], [0, max(y_test)], "r--", label="Optimal (y=x)")  # Ideal line

    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("Predicted vs Actual Distance")
    plt.legend()
    plt.grid()
    plt.show()


def main() -> None:
    df = pd.read_csv(DATASET_FILE_PATH)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # Shuffle dataset

    # Extract RSSI values as features (dropping 'location' and 'date')
    x_float = df.drop(columns=["location", "date"])

    # Transform RSSI values using a Tsetlin Machine-compatible binarizer
    binarizer = StandardBinarizer(max_bits_per_feature=10)
    x_transformed = binarizer.fit_transform(x_float.values)

    # Compute ground-truth distances to beacons
    y = []
    for _, row in df.iterrows():
        location_vector = location_key_to_vector_2d(row["location"])
        distances = [calc_distance(location_vector, beacon_pos) for beacon_pos in BEACON_POSITIONS.values()]
        y.append(distances)
    y = np.array(y)

    # Split dataset into training (80%) and testing (20%)
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x_transformed, y, df.index, test_size=0.2, random_state=1
    )

    # Train a separate Regression Tsetlin Machine (RTM) for each beacon
    rtms = []
    for i in range(len(BEACON_POSITIONS)):
        tm = TMRegressor(
            number_of_clauses=1000,
            T=5000,
            s=2.75,
            platform="CPU",
            weighted_clauses=True
        )

        y_train_beacon = y_train[:, i]
        y_test_beacon = y_test[:, i]

        for epoch in range(EPOCHS):
            tm.fit(x_train, y_train_beacon)
            rmse = np.sqrt(((tm.predict(x_test) - y_test_beacon) ** 2).mean())
            print(f"Beacon {i+1} - Epoch {epoch + 1}/{EPOCHS} - RMSE: {rmse:.2f}")
        
        rtms.append(tm)

    # Position estimation using Least Squares algorithm
    least_squares = LeastSquaresAlgorithm2D()
    distances_from_ground_truth = []
    y_actual = []
    y_predicted = []

    beacon_x_positions = [b.x for b in BEACON_POSITIONS.values()]
    beacon_y_positions = [b.y for b in BEACON_POSITIONS.values()]

    for idx in range(len(x_test)):
        sample_x = x_test[idx]
        test_index = test_indices[idx]
        predicted_distances = [rtms[i].predict(sample_x)[0] for i in range(len(BEACON_POSITIONS))]
        
        try:
            predicted_position = least_squares.predict(beacon_x_positions, beacon_y_positions, predicted_distances)
        except np.linalg.LinAlgError:
            print("Matrix is singular, skipping...")
            continue

        actual_position = location_key_to_vector_2d(df.loc[test_index, "location"])
        error_distance = calc_distance(predicted_position, actual_position)
        distances_from_ground_truth.append(error_distance)

        # Use idx to access the corresponding row in y_test
        y_actual.extend(y_test[idx])
        y_predicted.extend(predicted_distances)

    # Print evaluation metrics
    print(f"Average error: {np.mean(distances_from_ground_truth):.2f} m")
    print(f"Max error: {np.max(distances_from_ground_truth):.2f} m")
    print(f"Min error: {np.min(distances_from_ground_truth):.2f} m")
    print(f"Standard deviation: {np.std(distances_from_ground_truth):.2f} m")

    # Generate plots
    plot_rssi_vs_actual_distance(df)
    plot_predicted_vs_actual_distance(y_actual, y_predicted)


if __name__ == "__main__":
    main()
