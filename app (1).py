import streamlit as st
import pandas as pd
import numpy as np
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
        "🔍 Feature Importance",
        "🛠️ Maintenance Decision & Cost Analysis"
    ]
)


st.sidebar.markdown("---")

st.sidebar.write(
    """
    **Dataset**

    PHM Society 2010

    **Experiment**

    C1

    **Model**

    XGBoost Regression

    **Sensors**

    7

    **Features**

    63
    """
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


    # --------------------------------------------------------
    # AUTOMATIC INSIGHTS
    # --------------------------------------------------------

    st.subheader("💡 Key Insights")

    first_wear = df.sort_values(
        "Cut_Number"
    )["Tool_Wear"].iloc[0]

    last_wear = df.sort_values(
        "Cut_Number"
    )["Tool_Wear"].iloc[-1]

    increase = last_wear - first_wear


    st.info(
        f"🔧 Tool wear changed from **{first_wear:.2f}** "
        f"at the beginning to **{last_wear:.2f}** "
        f"towards the end."
    )

    st.info(
        f"📈 Overall wear increased by approximately "
        f"**{increase:.2f} wear units**."
    )

    st.info(
        f"🤖 The XGBoost model achieved an R² of "
        f"**{r2:.6f}**, indicating approximately "
        f"**{r2 * 100:.2f}% variance explained** "
        f"on the current test split."
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


    # --------------------------------------------------------
    # ERROR DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📉 Prediction Error Distribution"
    )


    error_df = pd.DataFrame({

        "Error":
            y_test.values -
            pred

    })


    fig_error = px.histogram(
        error_df,
        x="Error",
        nbins=20,
        title="Prediction Error Distribution"
    )


    st.plotly_chart(
        fig_error,
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
# ============================================================
# MAINTENANCE DECISION & COST ANALYSIS
# ============================================================

elif page == "🛠️ Maintenance Decision & Cost Analysis":

    st.header("🛠️ Maintenance Decision & Cost Analysis")

    st.write(
        "Estimate tool replacement requirements and maintenance costs "
        "using the predicted tool-wear values."
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

    st.markdown("---")

    # ========================================================
    # TOOL REPLACEMENT THRESHOLD
    # ========================================================

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

    # ========================================================
    # MAINTENANCE DECISION
    # ========================================================

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

    # ========================================================
    # COST ASSUMPTIONS
    # ========================================================

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

    # ========================================================
    # COST CALCULATION
    # ========================================================

    total_samples = len(y_test)

    baseline_cost = (
        total_samples * inspection_cost
    )

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
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Based Smart Tool Wear Prediction System | "
    "PHM Society 2010 Milling Data Challenge – C1 | "
    "XGBoost Regression"
)