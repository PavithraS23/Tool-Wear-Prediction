# ============================================================
# AI-BASED SMART TOOL WEAR PREDICTION SYSTEM
# PHM 2010 - C1 TOOL EXPERIMENT
#
# Pipeline:
# Sensor CSV files
#       ↓
# Data preprocessing
#       ↓
# Statistical feature extraction
#       ↓
# 63 features
#       ↓
# Tool wear target
#       ↓
# XGBoost Regression
#       ↓
# Prediction
#       ↓
# RMSE / MAE / R2
#       ↓
# SHAP Explainability
# ============================================================


import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

import shap

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "C1"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("AI-BASED SMART TOOL WEAR PREDICTION SYSTEM")
print("=" * 70)

print("\nData directory:")
print(DATA_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. SENSOR NAMES
# ============================================================

SENSOR_COLUMNS = [
    "Fx",
    "Fy",
    "Fz",
    "Vx",
    "Vy",
    "Vz",
    "AE"
]


# ============================================================
# 3. FIND SENSOR FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 - FINDING C1 SENSOR FILES")
print("=" * 70)


# Search for CSV files beginning with c_1_
sensor_files = glob.glob(
    os.path.join(DATA_DIR, "c_1_*.csv")
)

# Sort according to cut number
def get_cut_number(filepath):

    filename = os.path.basename(filepath)

    match = re.search(
        r"c_1_(\d+)",
        filename.lower()
    )

    if match:
        return int(match.group(1))

    return 999999


sensor_files = sorted(
    sensor_files,
    key=get_cut_number
)


print("\nNumber of sensor files found:")
print(len(sensor_files))


if len(sensor_files) == 0:

    print("\nERROR:")
    print("No c_1_*.csv files were found.")

    print("\nPlease check that your files are located here:")
    print(DATA_DIR)

    print("\nExample:")
    print("data/C1/c_1_001.csv")

    raise SystemExit


print("\nFirst 10 sensor files:")

for file in sensor_files[:10]:
    print(
        os.path.basename(file)
    )


print("\nLast 10 sensor files:")

for file in sensor_files[-10:]:
    print(
        os.path.basename(file)
    )


# ============================================================
# 4. CHECK CUT NUMBERS
# ============================================================

cut_numbers = [
    get_cut_number(file)
    for file in sensor_files
]

print("\nMinimum cut number:", min(cut_numbers))
print("Maximum cut number:", max(cut_numbers))

print(
    "Unique cut files:",
    len(set(cut_numbers))
)


# ============================================================
# 5. LOAD AND INSPECT ONE SENSOR FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - INSPECTING SENSOR FILE")
print("=" * 70)


sample_file = sensor_files[0]

print("\nSample file:")
print(
    os.path.basename(sample_file)
)


sample_data = pd.read_csv(
    sample_file,
    header=None
)


print("\nSample file shape:")
print(sample_data.shape)


print("\nFirst 5 rows:")
print(sample_data.head())


print("\nNumber of columns:")
print(sample_data.shape[1])


# ============================================================
# 6. VERIFY SENSOR COLUMNS
# ============================================================

if sample_data.shape[1] != 7:

    print("\nWARNING:")
    print(
        "Expected 7 sensor columns, "
        f"but found {sample_data.shape[1]}."
    )

    raise SystemExit


sample_data.columns = SENSOR_COLUMNS


print("\nSensor columns:")
print(SENSOR_COLUMNS)


# ============================================================
# 7. FEATURE EXTRACTION FUNCTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - DEFINING FEATURE EXTRACTION")
print("=" * 70)


def extract_features(data):

    features = {}

    for sensor in SENSOR_COLUMNS:

        # Convert to numeric
        signal = pd.to_numeric(
            data[sensor],
            errors="coerce"
        )

        # Remove NaN
        signal = signal.dropna()

        # Convert to NumPy
        signal = signal.to_numpy(
            dtype=float
        )

        # Remove infinity
        signal = signal[
            np.isfinite(signal)
        ]

        # Safety check
        if len(signal) == 0:

            features[
                f"{sensor}_Mean"
            ] = np.nan

            features[
                f"{sensor}_RMS"
            ] = np.nan

            features[
                f"{sensor}_Std"
            ] = np.nan

            features[
                f"{sensor}_Max"
            ] = np.nan

            features[
                f"{sensor}_Min"
            ] = np.nan

            features[
                f"{sensor}_Peak"
            ] = np.nan

            features[
                f"{sensor}_Variance"
            ] = np.nan

            features[
                f"{sensor}_Skewness"
            ] = np.nan

            features[
                f"{sensor}_Kurtosis"
            ] = np.nan

            continue


        # ----------------------------------------------------
        # 1. Mean
        # ----------------------------------------------------

        features[
            f"{sensor}_Mean"
        ] = np.mean(signal)


        # ----------------------------------------------------
        # 2. RMS
        # ----------------------------------------------------

        features[
            f"{sensor}_RMS"
        ] = np.sqrt(
            np.mean(
                signal ** 2
            )
        )


        # ----------------------------------------------------
        # 3. Standard Deviation
        # ----------------------------------------------------

        features[
            f"{sensor}_Std"
        ] = np.std(
            signal
        )


        # ----------------------------------------------------
        # 4. Maximum
        # ----------------------------------------------------

        features[
            f"{sensor}_Max"
        ] = np.max(
            signal
        )


        # ----------------------------------------------------
        # 5. Minimum
        # ----------------------------------------------------

        features[
            f"{sensor}_Min"
        ] = np.min(
            signal
        )


        # ----------------------------------------------------
        # 6. Peak
        # ----------------------------------------------------

        features[
            f"{sensor}_Peak"
        ] = np.max(
            np.abs(signal)
        )


        # ----------------------------------------------------
        # 7. Variance
        # ----------------------------------------------------

        features[
            f"{sensor}_Variance"
        ] = np.var(
            signal
        )


        # ----------------------------------------------------
        # 8. Skewness
        # ----------------------------------------------------

        features[
            f"{sensor}_Skewness"
        ] = pd.Series(
            signal
        ).skew()


        # ----------------------------------------------------
        # 9. Kurtosis
        # ----------------------------------------------------

        features[
            f"{sensor}_Kurtosis"
        ] = pd.Series(
            signal
        ).kurtosis()


    return features


# ============================================================
# 8. TEST FEATURE EXTRACTION
# ============================================================

test_features = extract_features(
    sample_data
)

print("\nNumber of extracted features:")
print(len(test_features))


if len(test_features) != 63:

    print(
        "\nWARNING: Expected 63 features."
    )

else:

    print(
        "\nSUCCESS: 63 features generated."
    )


print("\nFeature names:")

for feature in test_features.keys():

    print(feature)


# ============================================================
# 9. PROCESS ALL 315 SENSOR FILES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - PROCESSING ALL SENSOR FILES")
print("=" * 70)


feature_rows = []


for index, file_path in enumerate(
    sensor_files
):

    try:

        # Read CSV
        data = pd.read_csv(
            file_path,
            header=None
        )


        # Check number of columns
        if data.shape[1] != 7:

            print(
                f"\nSkipping {os.path.basename(file_path)}"
            )

            print(
                "Expected 7 columns, found:",
                data.shape[1]
            )

            continue


        # Assign sensor names
        data.columns = SENSOR_COLUMNS


        # Extract features
        features = extract_features(
            data
        )


        # Get cut number
        cut_number = get_cut_number(
            file_path
        )


        # Add cut number
        features[
            "Cut_Number"
        ] = cut_number


        # Add filename
        features[
            "File_Name"
        ] = os.path.basename(
            file_path
        )


        # Add to list
        feature_rows.append(
            features
        )


        # Progress
        if (
            (index + 1) % 25 == 0
            or
            index == 0
            or
            index == len(sensor_files) - 1
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(sensor_files)} files"
            )


    except Exception as e:

        print(
            f"\nERROR processing "
            f"{file_path}"
        )

        print(e)


# ============================================================
# 10. CREATE FEATURE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    feature_rows
)


print("\n" + "=" * 70)
print("STEP 5 - FEATURE DATASET CREATED")
print("=" * 70)


print(
    "\nFeature dataset shape:"
)

print(
    features_df.shape
)


print(
    "\nNumber of rows:",
    len(features_df)
)


# Sensor feature columns
feature_columns = [
    column
    for column in features_df.columns
    if column not in [
        "Cut_Number",
        "File_Name"
    ]
]


print(
    "\nNumber of sensor features:",
    len(feature_columns)
)


# ============================================================
# 11. SORT FEATURES BY CUT NUMBER
# ============================================================

features_df = features_df.sort_values(
    "Cut_Number"
).reset_index(
    drop=True
)


# ============================================================
# 12. CLEAN FEATURE DATA
# ============================================================

print("\nCleaning invalid values...")


features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Count missing values
missing_before = (
    features_df[
        feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "Missing values before cleaning:",
    missing_before
)


# Fill missing feature values
features_df[
    feature_columns
] = features_df[
    feature_columns
].fillna(
    features_df[
        feature_columns
    ].median()
)


missing_after = (
    features_df[
        feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


print(
    "Missing values after cleaning:",
    missing_after
)


# ============================================================
# 13. SAVE FEATURE DATASET
# ============================================================

feature_dataset_path = os.path.join(
    OUTPUT_DIR,
    "C1_feature_dataset.csv"
)


features_df.to_csv(
    feature_dataset_path,
    index=False
)


print(
    "\nFeature dataset saved:"
)

print(
    feature_dataset_path
)


# ============================================================
# 14. LOAD C1 WEAR FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - LOADING TOOL WEAR DATA")
print("=" * 70)


wear_file = os.path.join(
    DATA_DIR,
    "c1_wear.csv"
)


if not os.path.exists(
    wear_file
):

    print(
        "\nERROR: c1_wear.csv not found."
    )

    print(
        "Expected location:"
    )

    print(
        wear_file
    )

    raise SystemExit


wear_df = pd.read_csv(
    wear_file
)


print(
    "\nWear dataset shape:"
)

print(
    wear_df.shape
)


print(
    "\nWear columns:"
)

print(
    wear_df.columns.tolist()
)


print(
    "\nFirst 5 wear records:"
)

print(
    wear_df.head()
)


# ============================================================
# 15. VERIFY WEAR COLUMNS
# ============================================================

required_wear_columns = [
    "cut",
    "flute_1",
    "flute_2",
    "flute_3"
]


for column in required_wear_columns:

    if column not in wear_df.columns:

        print(
            f"\nERROR: Missing column: {column}"
        )

        print(
            "Available columns:",
            wear_df.columns.tolist()
        )

        raise SystemExit


# ============================================================
# 16. CREATE TOOL WEAR TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - CREATING TOOL WEAR TARGET")
print("=" * 70)


wear_df[
    "Tool_Wear"
] = (
    wear_df["flute_1"]
    +
    wear_df["flute_2"]
    +
    wear_df["flute_3"]
) / 3


print(
    "\nTool wear calculated as:"
)

print(
    "(flute_1 + flute_2 + flute_3) / 3"
)


print(
    "\nFirst 10 wear values:"
)

print(
    wear_df[
        [
            "cut",
            "flute_1",
            "flute_2",
            "flute_3",
            "Tool_Wear"
        ]
    ].head(10)
)


# ============================================================
# 17. MERGE SENSOR FEATURES WITH TOOL WEAR
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - MERGING SENSOR FEATURES + TOOL WEAR")
print("=" * 70)


# Rename wear cut column to Cut_Number
wear_merge_df = wear_df.rename(
    columns={
        "cut": "Cut_Number"
    }
)


final_df = pd.merge(
    features_df,
    wear_merge_df[
        [
            "Cut_Number",
            "flute_1",
            "flute_2",
            "flute_3",
            "Tool_Wear"
        ]
    ],
    on="Cut_Number",
    how="inner"
)


final_df = final_df.sort_values(
    "Cut_Number"
).reset_index(
    drop=True
)


print(
    "\nFinal dataset shape:"
)

print(
    final_df.shape
)


print(
    "\nFinal number of rows:",
    len(final_df)
)


print(
    "\nFirst 5 final records:"
)

print(
    final_df.head()
)


# ============================================================
# 18. CHECK DATA ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - DATA VALIDATION")
print("=" * 70)


print(
    "\nMinimum cut:",
    final_df["Cut_Number"].min()
)


print(
    "Maximum cut:",
    final_df["Cut_Number"].max()
)


print(
    "Number of matched cuts:",
    final_df["Cut_Number"].nunique()
)


print(
    "\nTool wear statistics:"
)

print(
    final_df["Tool_Wear"].describe()
)


# ============================================================
# 19. SAVE FINAL DATASET
# ============================================================

final_dataset_path = os.path.join(
    OUTPUT_DIR,
    "C1_final_dataset.csv"
)


final_df.to_csv(
    final_dataset_path,
    index=False
)


print(
    "\nFinal dataset saved:"
)

print(
    final_dataset_path
)


# ============================================================
# 20. PREPARE X AND Y
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - PREPARING MACHINE LEARNING DATA")
print("=" * 70)


X = final_df[
    feature_columns
]


y = final_df[
    "Tool_Wear"
]


print(
    "\nX shape:",
    X.shape
)


print(
    "y shape:",
    y.shape
)


print(
    "\nNumber of input features:",
    X.shape[1]
)


# ============================================================
# 21. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 22. CREATE XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - TRAINING XGBOOST REGRESSION MODEL")
print("=" * 70)


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


print(
    "\nTraining model..."
)


model.fit(
    X_train,
    y_train
)


print(
    "\nXGBoost model trained successfully!"
)


# ============================================================
# 23. GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - GENERATING FINAL PREDICTIONS")
print("=" * 70)


y_pred = model.predict(
    X_test
)


prediction_df = pd.DataFrame({

    "Cut_Number": final_df.loc[
        X_test.index,
        "Cut_Number"
    ].values,

    "Actual_Tool_Wear":
        y_test.values,

    "Predicted_Tool_Wear":
        y_pred

})


prediction_df[
    "Absolute_Error"
] = np.abs(
    prediction_df[
        "Actual_Tool_Wear"
    ]
    -
    prediction_df[
        "Predicted_Tool_Wear"
    ]
)


prediction_df = prediction_df.sort_values(
    "Cut_Number"
)


print(
    "\nFirst 15 predictions:"
)

print(
    prediction_df.head(15)
)


# ============================================================
# 24. CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


# ============================================================
# 25. CALCULATE MAE
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


# ============================================================
# 26. CALCULATE R2
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 27. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)


print(
    f"\nRMSE : {rmse:.6f}"
)


print(
    f"MAE  : {mae:.6f}"
)


print(
    f"R²   : {r2:.6f}"
)


print("\n" + "=" * 70)


# ============================================================
# 28. SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "C1_final_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


print(
    "\nPrediction file saved:"
)

print(
    prediction_path
)


# ============================================================
# 29. SAVE MODEL RESULTS
# ============================================================

results_df = pd.DataFrame({

    "Metric": [
        "RMSE",
        "MAE",
        "R2"
    ],

    "Value": [
        rmse,
        mae,
        r2
    ]

})


results_path = os.path.join(
    OUTPUT_DIR,
    "C1_model_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    "\nModel results saved:"
)

print(
    results_path
)


# ============================================================
# 30. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 - CREATING ACTUAL VS PREDICTED GRAPH")
print("=" * 70)


plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)


min_value = min(
    y_test.min(),
    y_pred.min()
)


max_value = max(
    y_test.max(),
    y_pred.max()
)


plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


plt.xlabel(
    "Actual Tool Wear"
)


plt.ylabel(
    "Predicted Tool Wear"
)


plt.title(
    "Actual vs Predicted Tool Wear"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)


plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 31. TOOL WEAR PROGRESSION GRAPH
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 - CREATING TOOL WEAR PROGRESSION GRAPH")
print("=" * 70)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    final_df["Cut_Number"],
    final_df["Tool_Wear"],
    marker="o",
    markersize=3
)


plt.xlabel(
    "Cut Number"
)


plt.ylabel(
    "Tool Wear"
)


plt.title(
    "Tool Wear Progression Across Cutting Operations"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


wear_progression_path = os.path.join(
    OUTPUT_DIR,
    "tool_wear_progression.png"
)


plt.savefig(
    wear_progression_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 32. XGBOOST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 - XGBOOST FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print(
    "\nTop 20 features:"
)

print(
    importance_df.head(20)
)


# ============================================================
# 33. FEATURE IMPORTANCE GRAPH
# ============================================================

top20 = importance_df.head(20)


plt.figure(
    figsize=(10, 7)
)


plt.barh(
    top20["Feature"][::-1],
    top20["Importance"][::-1]
)


plt.xlabel(
    "XGBoost Feature Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 20 Sensor Features"
)


plt.tight_layout()


feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_feature_importance.png"
)


plt.savefig(
    feature_importance_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 34. SHAP EXPLAINABILITY
# ============================================================

print("\n" + "=" * 70)
print("STEP 16 - SHAP EXPLAINABILITY")
print("=" * 70)


print(
    "\nCalculating SHAP values..."
)


explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    X_test
)


print(
    "SHAP values calculated successfully!"
)


# ============================================================
# 35. SHAP SUMMARY PLOT
# ============================================================

plt.figure()


shap.summary_plot(
    shap_values,
    X_test,
    max_display=20,
    show=False
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


plt.show()


# ============================================================
# 36. SHAP BAR PLOT
# ============================================================

plt.figure()


shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    max_display=20,
    show=False
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


plt.show()


# ============================================================
# 37. SAVE SHAP IMPORTANCE DATA
# ============================================================

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


shap_importance_path = os.path.join(
    OUTPUT_DIR,
    "SHAP_feature_importance.csv"
)


shap_importance_df.to_csv(
    shap_importance_path,
    index=False
)


# ============================================================
# 38. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nDataset:"
)

print(
    "PHM 2010 C1 Milling Dataset"
)


print(
    "\nSensor channels:"
)

print(
    "Fx, Fy, Fz, Vx, Vy, Vz, AE"
)


print(
    "\nFeatures per sensor:"
)

print(
    "9"
)


print(
    "\nTotal sensor features:"
)

print(
    "63"
)


print(
    "\nProcessed cuts:"
)

print(
    len(final_df)
)


print(
    "\nRMSE:"
)

print(
    f"{rmse:.6f}"
)


print(
    "\nMAE:"
)

print(
    f"{mae:.6f}"
)


print(
    "\nR²:"
)

print(
    f"{r2:.6f}"
)


print(
    "\nOutput files:"
)


print(
    "1. C1_feature_dataset.csv"
)


print(
    "2. C1_final_dataset.csv"
)


print(
    "3. C1_final_predictions.csv"
)


print(
    "4. C1_model_results.csv"
)


print(
    "5. actual_vs_predicted.png"
)


print(
    "6. tool_wear_progression.png"
)


print(
    "7. xgboost_feature_importance.png"
)


print(
    "8. shap_summary.png"
)


print(
    "9. shap_feature_importance.png"
)


print(
    "10. SHAP_feature_importance.csv"
)


print("\nAll files are inside:")
print(OUTPUT_DIR)


print("\n" + "=" * 70)