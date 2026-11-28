# AI-Based Smart Tool Wear Prediction System

An end-to-end predictive maintenance project that estimates cutting tool wear in milling from multi-sensor signals, with SHAP-based explainability and an interactive Streamlit dashboard.

Built with Python, Pandas, NumPy, XGBoost, SHAP and Streamlit as a team project (team of 3).

**Live demo:** [add your Streamlit app link here]

---

## Overview

Tool wear in milling affects part quality, machine downtime and cost. Replacing a tool too early wastes tool life, and replacing it too late damages parts. This project predicts the amount of tool wear directly from sensor signals recorded during cutting, so maintenance can be planned from data instead of fixed schedules.

The pipeline covers the full workflow: loading the sensor data, engineering statistical features, training and evaluating a regression model, explaining its predictions, and presenting the results in a dashboard.

---

## Dataset

- **Source:** PHM Society 2010 Data Challenge (milling dataset)
- **Samples:** 315 cuts
- **Signals:** 7 sensor channels covering cutting force, vibration and acoustic emission
- **Target:** measured tool wear for each cut

---

## Approach

1. **Data preparation:** load the raw sensor signals for each cut and organise them by channel.
2. **Feature engineering:** compute 9 statistical features per channel (including RMS, skewness and kurtosis), giving 63 features per cut.
3. **Modelling:** train an XGBoost regressor to predict tool wear from the 63 features.
4. **Evaluation:** measure performance with R2, RMSE and MAE on a random 80:20 train/test split.
5. **Explainability:** use SHAP to show which features drive each prediction.
6. **Dashboard:** present predictions, feature importance and wear progression in Streamlit.

---

## Results

| Metric | Value |
|---|---|
| R2 | 0.9977 |
| RMSE | 1.40 |
| MAE | 0.65 |

Evaluated on a random 80:20 test split (RMSE and MAE are in the units of the wear measurement).

### Key findings

- Vibration and cutting-force features were the strongest predictors of tool wear, for example `Vy_Std` and `Fx_Std`.
- SHAP analysis confirmed which signal features push the predicted wear higher or lower.

---

## Dashboard Features

- **Wear prediction:** predict tool wear from the sensor features
- **Feature importance:** see which features matter most, using SHAP
- **Wear progression:** view how wear develops across cuts
- **Maintenance cost comparison:** an illustrative comparison of maintenance strategies

---

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** XGBoost
- **Explainability:** SHAP
- **Dashboard:** Streamlit
- **Tools:** Jupyter Notebook, Git, GitHub

---

## Project Structure

> Adjust this to match your repository.

```
Smart-Tool-Wear-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
└── notebooks/
    └── tool_wear_prediction.ipynb
```

---

## Installation

Clone the repository:

```
git clone https://github.com/PavithraS23/<your-repo-name>.git

cd <your-repo-name>
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Run the Application

```
streamlit run app.py
```

---

## Limitations

- The model is evaluated on a random 80:20 split of one dataset. Consecutive cuts are highly correlated, so a random split can give optimistic scores.
- The dataset is small (315 cuts), so results may not carry over to other machines, tools or cutting conditions.
- The maintenance cost comparison is illustrative and not based on real plant data.

## Future Work

- Validate with a stricter split, such as holding out whole cutters or later cuts in time
- Test the approach on other tool wear datasets
- Compare XGBoost with other models and add cross-validation
- Add real-time prediction from live sensor data

---

## Authors

**Pavithra Senthilkumaran**

B.Tech – Artificial Intelligence and Data Science
Rajalakshmi Engineering College
