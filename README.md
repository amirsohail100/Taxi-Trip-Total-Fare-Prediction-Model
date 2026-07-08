# 🚖 Taxi Trip Total Fare Prediction (Multi-Model Ensemble)

An end-to-end Machine Learning regression project built to predict the `total` fare amount for taxi rides. This project evaluates and compares four powerful algorithms to find the most accurate and reliable architecture for fare forecasting.

---

## 📊 Model Performance & Evaluation ($R^2$ Scores)

The models were evaluated based on their coefficient of determination ($R^2$), delivering highly impressive results:

- **Linear Regression (LR):** **99.64% accuracy** (Best overall structural fit)
- **Gradient Boosting Regressor (GB):** **99.44% accuracy** (Exceptional ensemble learning)
- **XGBoost Regressor (XG):** **98.89% accuracy** (Highly optimized gradient boosting)
- **K-Neighbors Regressor (KNN):** **97.23% accuracy** (Robust distance-based baseline)

---

## 🚀 Key Features

- **Multi-Algorithm Benchmark:** Provides a transparent performance comparison across linear, distance-based, and tree-based ensemble models.
- **Exceptional Precision:** Achieves up to 99.64% accuracy by properly learning from features like trip distance, location patterns, and timestamps.
- **Production Ready:** The best-performing pipelines are serialized and structured for instant inference integration into ride-hailing software.

---

## 📁 Repository Structure

- `models/`: Directory containing serialized model files (`LR`, `GB`, `XG`, `KNN` saved via pickle/joblib).
- `preprocessor.pickle`: Serialized scaling and encoding pipeline for structural alignment.
- `train.py` / `notebook.ipynb`: Core Python code handling data cleaning, feature engineering, training, and evaluation.
- `README.md`: Comprehensive documentation.

---

## 💻 Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-Learn, XGBoost
- **Data Handling:** Pandas, NumPy

---

## 📜 License

This project is licensed under the MIT License.

A Machine Learning regression model built to predict the total fare amount for taxi trips. Trained on a comprehensive taxi dataset, it analyzes key features like trip distance, pickup/dropoff locations, passenger count, and time of day to deliver highly accurate total cost estimations for smart urban mobility.
