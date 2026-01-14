# 5G Network Slice Traffic Forecaster 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Skill-Deep_Learning_from_Scratch-orange)
![Data Science](https://img.shields.io/badge/Focus-Predictive_Analytics-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

> **End-to-end 5G network traffic simulation and demand prediction pipeline using a custom Deep Neural Network built from scratch (NumPy) and ensemble benchmarking.**

---

## About the Project

In the context of **5G Network Slicing**, dynamic resource allocation is critical for maintaining Quality of Service (QoS). This project simulates a 5G eMBB (Enhanced Mobile Broadband) slice environment and constructs a complete Machine Learning pipeline to predict **Throughput Demand (Mbps)** based on real-time network metrics.

**Core Objective:** Beyond prediction accuracy, this project serves as a technical proof-of-concept demonstrating the ability to **implement Deep Learning architecture mathematically from scratch** (without using TensorFlow/PyTorch for the core logic) and validating its performance against industry standards like **XGBoost**.

###  Key Competencies Demonstrated
* **Mathematical Deep Learning:** Manual implementation of Forward/Backward Propagation, Gradient Descent optimization, and Weight Initialization using pure linear algebra (NumPy).
* **Advanced Feature Engineering:** Implementation of Cyclical Feature Encoding (Sine/Cosine transformations) to correctly model temporal data (24-hour cycles).
* **Synthetic Data Generation:** Creation of a statistical generator using Beta and Poisson distributions to simulate realistic network stress, latency penalties, and user load.
* **Model Benchmarking:** Rigorous comparison between the custom DNN, Random Forest, and XGBoost to evaluate bias-variance trade-offs.

---

##  Architecture & Features

### 1. Synthetic 5G Data Generator
A custom script that models the behavior of a 5G cell:
* **Traffic Modeling:** Simulates time-dependent user peaks (e.g., evening congestion).
* **Complex Relations:** Non-linear dependencies between Latency (ms), Bandwidth (MHz), Packet Loss Rate, and Active Users.
* **Output:** A labeled dataset incorporating noise and anomalies to test model robustness.

### 2. Advanced Preprocessing (`clean_norm.py`)
* **Cyclical Time Encoding:** Transforms linear "Hour of Day" into 2D coordinates ($sin, cos$) to preserve continuity between 23:00 and 00:00.
* **Normalization:** Min-Max scaling applied to all input features to ensure efficient Gradient Descent convergence.

### 3.  Deep Neural Network (Custom Implementation)
A fully connected L-layer neural network built entirely in **NumPy**:
* **Architecture:** [Input: 6] $\rightarrow$ [FC: 20] $\rightarrow$ [FC: 7] $\rightarrow$ [FC: 5] $\rightarrow$ [Output: 1]
* **Optimization:** Gradient Descent with adaptive learning rates.
* **Performance:** Achieves ~0.90 $R^2$ Score on unseen test data.

### 4.  Comparative Analysis (XGBoost Integration)
Integration of `scikit-learn` and `xgboost` to benchmark the custom solution and analyze Feature Importance (explainability).

---

##  Prerequisites

* Python 3.8+
* Required Libraries:
    ```bash
    numpy
    pandas
    matplotlib
    scikit-learn
    xgboost
    ```

##  Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/5g-traffic-prediction.git](https://github.com/your-username/5g-traffic-prediction.git)
    cd 5g-traffic-prediction
    ```

2.  **Generate Data**
    Create the synthetic dataset `5g_slice_traffic.csv`:
    ```bash
    python traffic_generator.py
    ```

3.  **Train & Evaluate**
    Run the pipeline (cleaning, DNN training, and XGBoost comparison):
    ```bash
    python train_prediction.py
    ```

---

##  Results Snapshot

The project validates the custom Deep Learning implementation by comparing it against state-of-the-art regressors:

| Model | R² Score (Test) | Insights |
| :--- | :--- | :--- |
| **Custom DNN (NumPy)** | **0.9095** | High accuracy; validates correct implementation of backpropagation math. |
| **XGBoost** | ~0.9100 | Industry standard; slightly faster convergence on tabular data. |
| **Random Forest** | ~0.8900 | Robust baseline, but higher variance. |

> *Note: Cost convergence plots and Prediction vs. Reality graphs are automatically generated during training.*

---

##  NOTES

This project was developed to demonstrate full-stack Data Science capabilities: from statistical data generation and feature engineering to the low-level mathematical implementation of AI algorithms.



---
