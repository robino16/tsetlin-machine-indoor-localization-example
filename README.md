# Tsetlin Machine for Indoor Localization Example

This repository demonstrates how the [Tsetlin Machine](https://en.wikipedia.org/wiki/Tsetlin_machine) can be used for indoor localization using Bluetooth Low Energy (BLE) Received Signal Strength Indicator (RSSI) values. The dataset used is the [BLE RSSI Dataset for Indoor Localization](https://www.kaggle.com/datasets/mehdimka/ble-rssi-dataset) by Mehdi Mohammad on Kaggle.

## Overview

This is a minimal example of indoor localization using the **fingerprinting technique**. In real-world applications, fingerprinting requires manually collecting RSSI measurements at various locations, which is a time-consuming process.

### Key Findings:

- The approach achieves an **average localization error of ~2.1 meters**.
- In practical scenarios, real-world accuracy may be significantly worse due to various environmental factors and dataset limitations.

### Disclaimer:

This is an **oversimplified** example designed to demonstrate the use of the Tsetlin Machine for indoor localization. It **does not** represent a production-ready solution and has several limitations:

1. **Dataset Limitations:**

   - The exact methodology for collecting RSSI samples in the dataset is unclear.
   - It is assumed that the mobile device was accurately positioned within the predefined grid cells, but this is not explicitly verified.

2. **Potentially Unrealistic Data Splitting:**
   - The dataset is randomly split into **80% training and 20% testing**, which is a common approach in machine learning.
   - However, this may lead to **data leakage**, where identical or near-identical samples appear in both the training and test sets, making the model's performance seem better than it would be in real-world deployment.
   - A more realistic approach might involve **time-based splitting**, but even then, there is a risk of overlap between training and test samples.

For a more in-depth discussion of these challenges and potential improvements, see my **master's thesis**:  
[An Environment-Adaptive Approach for Indoor Localization Using the Tsetlin Machine](https://uia.brage.unit.no/uia-xmlui/handle/11250/2823874).

## How It Works

In this project, we use **multiple Regression Tsetlin Machines (RTMs)**—one for each access point (AP). Each RTM predicts the distance between the mobile device and the AP. This method provides more accurate distance estimations compared to a simple **path loss model**, which often struggles with environmental variations.

After predicting distances, we apply a localization algorithm:

- **Trilateration** (supports exactly 3 access points)
- **Least Squares Estimation** (supports 3 or more access points, used in this project)

The Least Squares method refines the estimated position by minimizing the error between predicted and actual distances.

---

## Prerequisites

Before running the code, ensure you have the following:

1. **Python** (>=3.8) – Download from [python.org](https://www.python.org)
2. **Tsetlin Machine (TMU)** – Install by following the instructions in the [TMU repository](https://github.com/cair/tmu)

## Dataset

The project utilizes the [BLE RSSI Dataset for Indoor Localization](https://www.kaggle.com/datasets/mehdimka/ble-rssi-dataset).
The dataset is licensed under **CC BY-NC-SA 4.0**, meaning it **cannot** be used for commercial purposes.

**⚠️ Dataset Not Included**

Due to licensing restrictions, we do not distribute the dataset in this repository.
You must download it manually from Kaggle and place it in the expected directory:

```
./data/iBeacon_RSSI_Labeled.csv
```

## Running the Project

To train and evaluate the Tsetlin Machine for indoor localization, run:

```sh
python main.py
```

This script:

- Loads the dataset
- Trains a Regression Tsetlin Machine for each access point
- Uses the Least Squares algorithm for final position estimation
- Outputs localization accuracy statistics

## Structure

```bash
├── main.py              # Loads data, trains models, evaluates localization accuracy
├── least_squares.py     # Implements Least Squares method for position estimation
├── common.py            # Helper classes (Vector2D, LocalizationAlgorithm2D)
├── data/                # Directory where the dataset should be placed (not included)
```

## Results

The script outputs localization error statistics, including:

- Average Error (meters)
- Maximum/Minimum Error
- Standard Deviation of Error

**Example output:**

```
Average error: 2.09 m
Max error: 10.03 m
Min error: 0.16 m
Standard deviation: 1.19 m
```

## Acknowledgments

- This project is based on the **Tsetlin Machine** and its implementation in the [TMU library](https://github.com/cair/tmu).
- The dataset was created by Mehdi Mohammad and is available on [Kaggle](https://www.kaggle.com/datasets/mehdimka/ble-rssi-dataset).
- The **Least Squares localization implementation** is based on the mathematical concepts described in:
  - **Ye, Z., Xu, Y., Lin, J., Li, G., Geng, E., & Pang, Y. (2018).**  
    ["An Improved Bluetooth Indoor Positioning Algorithm Based on RSSI and PSO-BPNN."](https://doi.org/10.3390/s18092820) _Sensors, 18(9), 2820._
  - This is **not a direct copy** but an independent implementation of the mathematical method presented in the paper.

## License & Usage

This project is for **educational and research purposes only**.
The dataset is CC BY-NC-SA 4.0, which means:

- ✔ You can use and modify the dataset for non-commercial purposes.
- ✔ You must credit the original author when using or referencing the dataset.
- ❌ You cannot use the dataset or derivative works for commercial purposes.

For full details, see the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
