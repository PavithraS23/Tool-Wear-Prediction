import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Tool Wear Prediction",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    font-size: 17px;
    color: #777;
    margin-bottom: 25px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">⚙️ AI-Based Smart Tool Wear Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered monitoring and prediction of cutting tool wear'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "outputs/C1_final_dataset.csv"


@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


try:

    df = load_data()

except Exception as e:

    st.error("Unable to load the dataset.")

    st.code(str(e))

    st.stop()


# ============================================================
# FEATURE COLUMNS
# ============================================================

excluded = [
    "Cut_Number",
    "File_Name",
    "flute_1",
    "flute_2",
    "flute_3",
    "Tool_Wear"
]

feature_columns = [
    c for c in df.columns
    if c not in excluded
]


X = df[feature_columns]
y = df["Tool_Wear"]


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

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

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    return (
        model,
        X_test,
        y_test,
        pred,
        mae,
        rmse,
        r2
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Tool Wear Monitor")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Model Performance",
        "📈 Wear Analysis",
        "🛠️ Maintenance Decision & Cost Analysis",
        "🔍 Feature Importance",
        "🧠 Explainable AI (SHAP)",
    ]
)




# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("📊 System Overview")

    st.write(
        "Overview of the tool-wear prediction system and dataset."
    )

    # Train only when needed
    with st.spinner("Initializing prediction model..."):

        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Cuts",
            len(df)
        )

    with c2:

        st.metric(
            "Sensor Channels",
            7
        )

    with c3:

        st.metric(
            "Extracted Features",
            63
        )

    with c4:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader("🎯 Model Performance")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "MAE",
            f"{mae:.4f}"
        )

    with m2:

        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

    with m3:

        st.metric(
            "R²",
            f"{r2:.6f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # TOOL WEAR TREND
    # --------------------------------------------------------

    st.subheader("📈 Tool Wear Progression")

    fig = px.line(
        df.sort_values("Cut_Number"),
        x="Cut_Number",
        y="Tool_Wear",
        markers=True,
        title="Tool Wear Across Cutting Operations"
    )

    fig.update_layout(
        xaxis_title="Cut Number",
        yaxis_title="Tool Wear"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.header("🔮 Tool Wear Prediction")

    st.write(
        "Select a cutting operation to estimate its tool wear."
    )


    # Train model

    with st.spinner("Loading prediction model..."):

        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()


    # --------------------------------------------------------
    # SELECT CUT
    # --------------------------------------------------------

    cut_numbers = df["Cut_Number"].tolist()

    selected_cut = st.selectbox(
        "Select Cut Number",
        cut_numbers
    )


    selected_row = df[
        df["Cut_Number"] == selected_cut
    ].iloc[0]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            selected_row[feature_columns].values
        ],
        columns=feature_columns
    )


    predicted_wear = model.predict(
        input_data
    )[0]


    actual_wear = selected_row[
        "Tool_Wear"
    ]


    error = abs(
        actual_wear -
        predicted_wear
    )


    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    max_wear = df[
        "Tool_Wear"
    ].max()


    wear_percentage = (
        predicted_wear /
        max_wear
    ) * 100


    if wear_percentage < 40:

        condition = "🟢 HEALTHY"

        recommendation = (
            "Tool condition is currently healthy. "
            "Continue normal monitoring."
        )

    elif wear_percentage < 70:

        condition = "🟡 MODERATE WEAR"

        recommendation = (
            "Tool wear is increasing. "
            "Continue close monitoring."
        )

    elif wear_percentage < 90:

        condition = "🟠 HIGH WEAR"

        recommendation = (
            "High tool wear detected. "
            "Consider scheduling tool inspection."
        )

    else:

        condition = "🔴 CRITICAL WEAR"

        recommendation = (
            "Critical tool wear detected. "
            "Tool replacement should be considered."
        )


    # --------------------------------------------------------
    # RESULT CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Cut Number",
            selected_cut
        )

    with c2:

        st.metric(
            "Actual Wear",
            f"{actual_wear:.2f}"
        )

    with c3:

        st.metric(
            "Predicted Wear",
            f"{predicted_wear:.2f}"
        )

    with c4:

        st.metric(
            "Prediction Error",
            f"{error:.2f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    st.subheader("🛠️ Tool Condition")

    st.success(condition)

    st.info(
        recommendation
    )


    # --------------------------------------------------------
    # GAUGE
    # --------------------------------------------------------

    st.subheader("📊 Predicted Wear Level")

    gauge = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=predicted_wear,

            title={
                "text": "Tool Wear"
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        max_wear
                    ]
                }
            }

        )
    )

    gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FLUTE WEAR
    # --------------------------------------------------------

    st.subheader(
        "🔧 Individual Flute Wear"
    )


    flute_df = pd.DataFrame({

        "Flute": [
            "Flute 1",
            "Flute 2",
            "Flute 3"
        ],

        "Wear": [
            selected_row["flute_1"],
            selected_row["flute_2"],
            selected_row["flute_3"]
        ]

    })


    fig = px.bar(
        flute_df,
        x="Flute",
        y="Wear",
        title="Wear Distribution Across Flutes"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    with st.spinner("Loading model evaluation..."):

        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "MAE",
            f"{mae:.6f}"
        )

    with c2:

        st.metric(
            "RMSE",
            f"{rmse:.6f}"
        )

    with c3:

        st.metric(
            "R²",
            f"{r2:.6f}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Tool Wear"
    )


    result_df = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": pred

    })


    fig = px.scatter(
        result_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Tool Wear"
    )


    minimum = min(
        result_df["Actual"].min(),
        result_df["Predicted"].min()
    )


    maximum = max(
        result_df["Actual"].max(),
        result_df["Predicted"].max()
    )


    fig.add_trace(
        go.Scatter(

            x=[
                minimum,
                maximum
            ],

            y=[
                minimum,
                maximum
            ],

            mode="lines",

            name="Ideal Prediction"

        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# WEAR ANALYSIS
# ============================================================

elif page == "📈 Wear Analysis":

    st.header("📈 Tool Wear Analysis")


    # --------------------------------------------------------
    # WEAR TREND
    # --------------------------------------------------------

    fig = px.line(

        df.sort_values(
            "Cut_Number"
        ),

        x="Cut_Number",

        y="Tool_Wear",

        markers=True,

        title="Tool Wear Progression"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FLUTE COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "🔧 Flute Wear Comparison"
    )


    flute_long = df[
        [
            "Cut_Number",
            "flute_1",
            "flute_2",
            "flute_3"
        ]
    ].melt(

        id_vars=[
            "Cut_Number"
        ],

        var_name="Flute",

        value_name="Wear"

    )


    fig2 = px.line(

        flute_long,

        x="Cut_Number",

        y="Wear",

        color="Flute",

        title="Individual Flute Wear Progression"

    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Wear Statistics"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Minimum",
            f"{df['Tool_Wear'].min():.2f}"
        )


    with c2:

        st.metric(
            "Maximum",
            f"{df['Tool_Wear'].max():.2f}"
        )


    with c3:

        st.metric(
            "Average",
            f"{df['Tool_Wear'].mean():.2f}"
        )


    with c4:

        st.metric(
            "Std. Deviation",
            f"{df['Tool_Wear'].std():.2f}"
        )


# ============================================================
# MAINTENANCE DECISION & COST ANALYSIS
# ============================================================

elif page == "🛠️ Maintenance Decision & Cost Analysis":

    st.header("🛠️ Maintenance Decision & Cost Analysis")

    st.write(
        "Estimate tool replacement requirements and maintenance costs "
        "using the predicted tool-wear values."
    )

    with st.spinner("Loading prediction model..."):
        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()

    st.markdown("---")

    st.subheader("🔧 Tool Replacement Threshold")

    threshold = st.number_input(
        "Enter tool wear replacement threshold",
        min_value=float(y_test.min()),
        max_value=float(y_test.max()),
        value=float(y.quantile(0.80)),
        step=1.0
    )

    st.caption(
        "The threshold is application-defined and should be validated "
        "according to the actual tool-wear measurement and machining process."
    )

    actual_replace = (
        y_test.to_numpy() >= threshold
    ).astype(int)

    predicted_replace = (
        np.asarray(pred) >= threshold
    ).astype(int)

    replacement_count = int(predicted_replace.sum())
    monitoring_count = int(len(predicted_replace) - replacement_count)

    st.subheader("📋 Maintenance Recommendation")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Tools Recommended for Replacement",
            replacement_count
        )

    with c2:
        st.metric(
            "Tools Recommended for Monitoring",
            monitoring_count
        )

    if replacement_count > 0:
        st.warning(
            f"{replacement_count} tool(s) have predicted wear above "
            f"the selected threshold."
        )
    else:
        st.success(
            "No tools have predicted wear above the selected threshold."
        )

    st.markdown("---")

    st.subheader("💰 Maintenance Cost Estimation")

    st.info(
        "The following costs are example assumptions. "
        "Update them using actual industrial maintenance costs."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        replacement_cost = st.number_input(
            "Tool replacement cost (₹)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

    with c2:
        inspection_cost = st.number_input(
            "Manual inspection cost (₹)",
            min_value=0.0,
            value=150.0,
            step=25.0
        )

    with c3:
        monitoring_cost = st.number_input(
            "Monitoring cost per sample (₹)",
            min_value=0.0,
            value=2.0,
            step=1.0
        )

    total_samples = len(y_test)

    baseline_cost = total_samples * inspection_cost

    proposed_cost = (
        replacement_count * replacement_cost
        + total_samples * monitoring_cost
    )

    cost_difference = baseline_cost - proposed_cost

    st.subheader("📊 Cost Comparison")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Manual Inspection Cost",
            f"₹{baseline_cost:,.2f}"
        )

    with c2:
        st.metric(
            "Proposed System Cost",
            f"₹{proposed_cost:,.2f}"
        )

    with c3:
        st.metric(
            "Estimated Cost Difference",
            f"₹{cost_difference:,.2f}"
        )

    cost_df = pd.DataFrame({
        "Method": [
            "Manual Inspection",
            "Proposed System"
        ],
        "Cost": [
            baseline_cost,
            proposed_cost
        ]
    })

    fig = px.bar(
        cost_df,
        x="Method",
        y="Cost",
        title="Estimated Maintenance Cost Comparison",
        text="Cost"
    )

    fig.update_layout(
        yaxis_title="Estimated Cost (₹)",
        xaxis_title="Maintenance Method"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "This cost comparison is an assumption-based simulation "
        "and does not represent validated industrial cost savings."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.header(
        "🔍 XGBoost Feature Importance"
    )


    with st.spinner(
        "Calculating feature importance..."
    ):

        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()


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


    top_n = st.slider(
        "Number of features",
        5,
        30,
        15
    )


    top_features = importance_df.head(
        top_n
    )


    fig = px.bar(

        top_features.sort_values(
            "Importance"
        ),

        x="Importance",

        y="Feature",

        orientation="h",

        title=f"Top {top_n} Important Features"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "📋 Feature Importance Table"
    )


    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPLAINABLE AI - SHAP
# ============================================================

elif page == "🧠 Explainable AI (SHAP)":

    st.header("🧠 Explainable AI – SHAP")
    st.write(
        "Understand why the XGBoost model predicted a particular tool-wear "
        "value and which sensor-derived features pushed the prediction higher "
        "or lower."
    )

    # SHAP is loaded only when this page is opened so the rest of the
    # application can still load if the package has not been installed.
    try:
        import shap
    except ImportError:
        st.error(
            "SHAP is not installed. Install it with: pip install shap"
        )
        st.stop()

    with st.spinner("Preparing SHAP explanation..."):
        (
            model,
            X_test,
            y_test,
            pred,
            mae,
            rmse,
            r2
        ) = train_model()

    st.markdown("---")

    # --------------------------------------------------------
    # SELECT CUT
    # --------------------------------------------------------

    st.subheader("🔮 Select a Cutting Operation")

    cut_numbers = df["Cut_Number"].tolist()

    selected_shap_cut = st.selectbox(
        "Select Cut Number to Explain",
        cut_numbers,
        key="shap_cut"
    )

    selected_shap_row = df[
        df["Cut_Number"] == selected_shap_cut
    ].iloc[0]

    shap_input = pd.DataFrame(
        [selected_shap_row[feature_columns].values],
        columns=feature_columns
    )

    predicted_shap_wear = float(model.predict(shap_input)[0])
    actual_shap_wear = float(selected_shap_row["Tool_Wear"])

    # --------------------------------------------------------
    # SHAP VALUES
    # --------------------------------------------------------

    # Use SHAP's model-agnostic permutation explainer instead of
    # TreeExplainer. This avoids a compatibility issue between some
    # XGBoost versions and SHAP where XGBoost's base_score can be stored
    # as a vector string such as "[1.0862423E2]".
    #
    # A small background sample keeps the explanation responsive while
    # still providing a meaningful local explanation for the selected cut.
    background = X_test.iloc[:min(50, len(X_test))].copy()

    explainer = shap.Explainer(
        model.predict,
        background,
        algorithm="permutation"
    )

    max_evals = max(
        2 * len(feature_columns) + 1,
        500
    )

    shap_result = explainer(
        shap_input,
        max_evals=max_evals
    )

    shap_values = np.asarray(shap_result.values).reshape(-1)

    # For a callable/model-agnostic explainer, the base value is the
    # expected model output for the background data.
    base_value = float(np.asarray(shap_result.base_values).reshape(-1)[0])

    explanation_df = pd.DataFrame({
        "Feature": feature_columns,
        "Feature Value": shap_input.iloc[0].values,
        "SHAP Value": shap_values,
        "Absolute SHAP": np.abs(shap_values)
    }).sort_values(
        "Absolute SHAP",
        ascending=False
    )

    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Actual Tool Wear",
            f"{actual_shap_wear:.2f}"
        )

    with c2:
        st.metric(
            "Predicted Tool Wear",
            f"{predicted_shap_wear:.2f}"
        )

    with c3:
        st.metric(
            "Model Baseline",
            f"{base_value:.2f}"
        )

    st.info(
        "SHAP interpretation: a positive SHAP value pushes the prediction "
        "towards higher tool wear, while a negative SHAP value pushes it "
        "towards lower tool wear."
    )

    # --------------------------------------------------------
    # TOP FEATURES AFFECTING THIS PREDICTION
    # --------------------------------------------------------

    st.subheader("🎯 What Influenced This Prediction?")

    top_n_shap = st.slider(
        "Number of contributing features",
        min_value=5,
        max_value=min(20, len(feature_columns)),
        value=min(10, len(feature_columns)),
        key="shap_top_n"
    )

    # IMPORTANT: Do not select the top features only by absolute SHAP and
    # then expect both positive and negative directions to appear. A cut can
    # legitimately have its strongest contributors all on one side.
    # Instead, explicitly take the strongest positive and strongest negative
    # contributors so the user can see BOTH: what increased wear and what
    # reduced wear.
    positive_features = explanation_df[
        explanation_df["SHAP Value"] > 0
    ].nlargest(
        top_n_shap,
        "SHAP Value"
    ).copy()

    negative_features = explanation_df[
        explanation_df["SHAP Value"] < 0
    ].nsmallest(
        top_n_shap,
        "SHAP Value"
    ).copy()

    st.markdown("### 🔴 Readings Increasing Predicted Wear")

    if len(positive_features) > 0:
        positive_plot = positive_features.sort_values("SHAP Value")

        fig_positive = px.bar(
            positive_plot,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            title=f"Features Pushing Cut {selected_shap_cut} Toward Higher Wear",
            hover_data={
                "Feature Value": ":.4f",
                "SHAP Value": ":.4f"
            }
        )
        fig_positive.update_traces(marker_color="#ff5c5c")
        fig_positive.update_layout(
            xaxis_title="SHAP Value → Increase in Predicted Wear",
            yaxis_title="Feature",
            height=max(350, len(positive_plot) * 38)
        )
        st.plotly_chart(fig_positive, use_container_width=True)
    else:
        st.success(
            "No positive SHAP contributors were found for this cut. "
            "The model is not being pushed toward higher wear by any input "
            "feature for this particular operation."
        )

    st.markdown("### 🟢 Readings Reducing Predicted Wear")

    if len(negative_features) > 0:
        negative_plot = negative_features.sort_values("SHAP Value")

        fig_negative = px.bar(
            negative_plot,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            title=f"Features Pushing Cut {selected_shap_cut} Toward Lower Wear",
            hover_data={
                "Feature Value": ":.4f",
                "SHAP Value": ":.4f"
            }
        )
        fig_negative.update_traces(marker_color="#57d68d")
        fig_negative.update_layout(
            xaxis_title="SHAP Value → Decrease in Predicted Wear",
            yaxis_title="Feature",
            height=max(350, len(negative_plot) * 38)
        )
        st.plotly_chart(fig_negative, use_container_width=True)
    else:
        st.info(
            "No negative SHAP contributors were found for this cut."
        )

    # --------------------------------------------------------
    # SIMPLE EXPLANATION FOR THE END USER
    # --------------------------------------------------------

    st.subheader("💡 Why does the model think the tool is wearing?")

    st.write(
        "This converts the model explanation into simple language. "
        "It shows which sensor-derived readings had the biggest influence "
        "on the predicted tool wear for this cutting operation."
    )

    explanation_df = explanation_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )

    # For an end-user question such as "Why is the tool wearing?",
    # prioritize features that actually push the prediction upward.
    positive_reasons = explanation_df[
        explanation_df["SHAP Value"] > 0
    ].nlargest(3, "SHAP Value").copy()

    negative_reasons = explanation_df[
        explanation_df["SHAP Value"] < 0
    ].nsmallest(3, "SHAP Value").copy()

    top_reasons = positive_reasons.copy()
    if len(top_reasons) == 0:
        top_reasons = negative_reasons.copy()

    def readable_feature_name(name):
        text = str(name).replace("_", " ").replace("-", " ")
        text = re.sub(r"\\s+", " ", text).strip()
        return text.title()

    positive_count = int((explanation_df["SHAP Value"] > 0).sum())
    negative_count = int((explanation_df["SHAP Value"] < 0).sum())

    if predicted_shap_wear > base_value and positive_count > 0:
        overall_message = (
            "The prediction is above the model baseline. The positive "
            "contributors shown below are the main readings pushing the "
            "predicted tool wear higher."
        )
    elif predicted_shap_wear < base_value:
        overall_message = (
            "The prediction is below the model baseline for this cut. "
            "The negative contributors are pushing the predicted wear lower; "
            "the model is not saying that every reading indicates increased wear."
        )
    elif positive_count > 0:
        overall_message = (
            "The prediction is around the model baseline. Several readings "
            "are pushing it upward while other readings push it downward."
        )
    else:
        overall_message = (
            "The model prediction is not being pushed upward by any positive "
            "SHAP contributor for this particular cut."
        )

    st.info(
        f"🔎 **Prediction explanation:** {overall_message}"
    )

    st.markdown("### What is influencing the prediction?")

    for _, reason in top_reasons.iterrows():

        feature_name = readable_feature_name(reason["Feature"])
        feature_value = reason["Feature Value"]
        shap_value = float(reason["SHAP Value"])

        if shap_value > 0:
            direction = "increasing the predicted wear"
            icon = "🔴"
        elif shap_value < 0:
            direction = "reducing the predicted wear"
            icon = "🟢"
        else:
            direction = "having almost no effect on the prediction"
            icon = "⚪"

        try:
            value_text = f"{float(feature_value):.3f}"
        except (TypeError, ValueError):
            value_text = str(feature_value)

        st.markdown(
            f"""
            <div style="
                padding:16px;
                margin:8px 0;
                border-radius:10px;
                background:#172b40;
                border-left:4px solid #4da3ff;
            ">
                <b>{icon} {feature_name}</b><br>
                Sensor-derived reading: <b>{value_text}</b><br>
                This reading is <b>{direction}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

    if predicted_shap_wear > actual_shap_wear:
        comparison_text = (
            "The model's estimated wear is higher than the measured wear "
            "for this cut."
        )
    elif predicted_shap_wear < actual_shap_wear:
        comparison_text = (
            "The model's estimated wear is lower than the measured wear "
            "for this cut."
        )
    else:
        comparison_text = (
            "The model's estimated wear is very close to the measured wear "
            "for this cut."
        )

    st.success(
        f"🛠️ **In simple terms:** The model looks at the sensor-derived "
        f"readings from this cutting operation and combines their learned "
        f"patterns to estimate tool wear. The readings listed above are the "
        f"ones that influenced this particular prediction the most. "
        f"{comparison_text}"
    )

    st.caption(
        "SHAP explains why the model made this prediction from the available "
        "input features. It does not by itself prove that one sensor "
        "physically caused the tool wear."
    )

    # --------------------------------------------------------
    # DETAILED EXPLANATION TABLE
    # --------------------------------------------------------

   


