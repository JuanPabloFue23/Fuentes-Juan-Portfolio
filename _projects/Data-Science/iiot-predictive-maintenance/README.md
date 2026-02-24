# 🏭 Industrial IoT: Predictive Maintenance Pipeline


This project implements a **production-grade Machine Learning pipeline** to predict industrial equipment failure. Instead of using static datasets, it features a custom **Synthetic Data Engine** that simulates physical sensor interactions and realistic class imbalances.

## 🎯 Project Highlights
* **Synthetic Data Generation:** Simulates physics-based failures (Tool Wear × Torque interaction) using `Numpy` and `Faker`.
* **Modular Architecture:** Separate modules for generation, feature engineering, and training to ensure maintainability.
* **Production-Ready Preprocessing:** Uses `scikit-learn` `ColumnTransformer` to prevent data leakage and ensure reproducibility.
* **Config-Driven:** All hyperparameters and file paths are managed via a central `config.yaml`.

## 🏗 Project Structure
```text
project/
├── config.yaml          # Central configuration source (Single Source of Truth)
├── data/                # Local data storage (Ignored by Git)
├── models/              # Serialized pipeline (.joblib)
├── reports/             # Evaluation artifacts (PR Curves, Confusion Matrix)
├── src/                 # Source code
│   ├── data_gen.py      # Synthetic engine & physics logic
│   ├── preprocessing.py # Scaling, encoding, and pipeline definition
│   ├── features.py      # Domain-specific feature engineering
│   ├── train.py         # Champion vs. Challenger training logic
│   └── evaluate.py      # Performance deep-dives and visualizations
└── requirements.txt     # Pinned dependencies for reproducibility

## 📈 Model Performance
We prioritize Precision-Recall AUC over Accuracy due to the inherent class imbalance (5% failure rate). By optimizing the decision threshold, the model is designed to minimize False Negatives—preventing costly machine downtime while managing the rate of unnecessary maintenance checks.