🚗 Vehicle Value AI: Advanced Price Prediction System

1.0 📌 Project Overview

Vehicle Value AI is a production-grade machine learning application designed to estimate used vehicle market values with $\mathbf{95.72\%}$ accuracy. Built on a robust stack of XGBoost, Scikit-Learn, and Streamlit, this system leverages advanced feature engineering and a scalable pipeline architecture to deliver real-time, data-driven valuations.

Unlike standard regression models, this system incorporates depreciation curves, usage intensity metrics, and hierarchical categorical encoding to capture the nuanced non-linear relationships in automotive pricing:

$$P_{final} = f(\text{Age}_{depreciation}, \text{Usage}_{intensity}, \text{Category}_{encoded})$$

2.0 🚀 Key Technical Innovations

2.1 Production-Grade ML Pipeline

Encapsulated Logic: Utilizes sklearn.pipeline.Pipeline to bundle preprocessing (imputation, encoding, scaling) and inference into a single, portable artifact (.pkl).

Zero Training-Serving Skew: Ensures that transformations $T(x)$ applied during training are identical to those during inference: 

$$T(x_{train}) \equiv T(x_{inference})$$

2.2 Advanced Feature Engineering

Depreciation Dynamics: Dynamically calculates $\Delta t$ (Vehicle Age) and applies non-linear decay models.

Usage Intensity: Computes $\frac{\text{Miles}}{\text{Year}}$ to differentiate between "highway miles" (high usage, lower wear) and city driving.

High-Cardinality Handling: Implements Target Encoding for Model and Trim ($1000+$ categories), enabling the model to learn from granular variations without memory explosion $O(1)$.

2.3 Optimized XGBoost Regressor

Histogram-Based Training: Leverages tree_method='hist' for $O(n)$ training complexity, allowing efficient scaling to $1\text{M}+$ rows.

Robustness: Tuned hyperparameters ($\eta = 0.02$, $\text{max\_depth} = 9$) balance bias and variance, achieving a sub-$\mathbf{\$830}$ Mean Absolute Error.

3.0 📊 Model Performance

The model was rigorously evaluated on a held-out test set ($20\%$) of $200,000$ records.

Metric

Value

Formula / Interpretation

Accuracy

$\mathbf{95.72\%}$

$1 - \text{MAPE}$ (Mean Absolute Percentage Error)

R² Score

$\mathbf{0.9917}$

$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ (Variance Explained)

MAE

$\mathbf{\$829.68}$

$\frac{1}{n}\sum_{i=1}^n

RMSE

$\mathbf{\$1,245.51}$

$\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}$ (Outlier Penalty)

Insight: Correlation analysis confirmed Vehicle Age ($\rho = -0.66$) and Engine HP ($\rho = +0.65$) as the primary drivers of value.

4.0 🏗️ System Architecture

graph LR
    A[Raw Data Ingestion] --> B{Feature Engineering}
    B -->|Calculate| C[Depreciation & Usage Metrics]
    B -->|Impute| D[Missing Values]
    C --> E{Scikit-Learn Pipeline}
    D --> E
    E -->|Target Encoding| F[High Cardinality Features]
    E -->|One-Hot| G[Low Cardinality Features]
    E -->|Scaling| H[Numeric Features]
    F --> I[XGBoost Regressor]
    G --> I
    H --> I
    I --> J[Streamlit Dashboard]


5.0 🛠️ Tech Stack

Core: Python 3.11, Pandas, NumPy

Modeling: XGBoost (Gradient Boosting), Scikit-Learn

Encoders: Category Encoders (Target Encoder), Joblib

Frontend: Streamlit (Custom CSS & Components)

Deployment: Streamlit Cloud (CI/CD from GitHub)

6.0 📂 Repository Structure

├── models/                     # Serialized Model Artifacts
│   ├── vehicle_price_pipeline.pkl  # The Brain: Full Preprocessing + XGBoost Model
│   └── input_columns.pkl           # The Map: Schema validation list
├── src/                        # Development & Training
│   └── production_train.py         # Advanced training script (Run to regenerate models)
├── app.py                      # Streamlit Application Entry Point
├── requirements.txt            # Python Dependencies
├── packages.txt                # System Dependencies (libgomp1)
└── README.md                   # Project Documentation


7.0 ⚡ Quick Start

7.1 Clone the Repository

git clone [https://github.com/YOUR_USERNAME/vehicle-price-predictor.git](https://github.com/YOUR_USERNAME/vehicle-price-predictor.git)
cd vehicle-price-predictor


7.2 Install Dependencies

pip install -r requirements.txt


7.3 Run the App Locally

streamlit run app.py
