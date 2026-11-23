import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

FINAL_DATASET = os.path.join(
    OUTPUT_DIR,
    "C1_final_dataset.csv"
)


print("=" * 70)
print("SHAP EXPLAINABILITY GENERATION")
print("=" * 70)


# ============================================================
# 2. LOAD FINAL DATASET
# ============================================================

print("\nLoading final dataset...")

df = pd.read_csv(
    FINAL_DATASET
)

print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

exclude_columns = [
    "Cut_Number",
    "File_Name",
    "flute_1",
    "flute_2",
    "flute_3",
    "Tool_Wear"
]


feature_columns = [
    col
    for col in df.columns
    if col not in exclude_columns
]


X = df[
    feature_columns
]

y = df[
    "Tool_Wear"
]


print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. TRAIN XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")

model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("XGBoost model trained successfully!")


# ============================================================
# 6. CREATE SHAP EXPLAINER
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_test
)


print("SHAP values calculated successfully!")


# ============================================================
# 7. SHAP SUMMARY PLOT
# ============================================================

print("\nGenerating SHAP summary plot...")

plt.figure(
    figsize=(10, 8)
)

shap.summary_plot(
    shap_values,
    X_test,
    max_display=20,
    show=False
)

plt.title(
    "SHAP Summary Plot - Tool Wear Prediction"
)

plt.tight_layout()


shap_summary_path = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)


plt.savefig(
    shap_summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    shap_summary_path
)


# ============================================================
# 8. SHAP BAR PLOT
# ============================================================

print("\nGenerating SHAP feature importance plot...")

plt.figure(
    figsize=(10, 8)
)

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title(
    "SHAP Feature Importance - Tool Wear Prediction"
)

plt.tight_layout()


shap_bar_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)


plt.savefig(
    shap_bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    shap_bar_path
)


# ============================================================
# 9. CALCULATE NUMERICAL SHAP IMPORTANCE
# ============================================================

print("\nCalculating SHAP feature importance values...")

mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)


shap_importance_df = pd.DataFrame({

    "Feature":
        X_test.columns,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})


shap_importance_df = shap_importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


# ============================================================
# 10. SAVE SHAP CSV
# ============================================================

shap_csv_path = os.path.join(
    OUTPUT_DIR,
    "SHAP_feature_importance.csv"
)


shap_importance_df.to_csv(
    shap_csv_path,
    index=False
)


print(
    "Saved:",
    shap_csv_path
)


# ============================================================
# 11. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES BASED ON SHAP")
print("=" * 70)

print(
    shap_importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 12. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNew files created:")

print(
    "1. shap_summary.png"
)

print(
    "2. shap_feature_importance.png"
)

print(
    "3. SHAP_feature_importance.csv"
)

print("\nLocation:")

print(
    OUTPUT_DIR
)