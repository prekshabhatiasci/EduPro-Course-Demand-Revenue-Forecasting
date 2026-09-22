# ============================================================
# EDUPRO
# COURSE DEMAND & REVENUE FORECASTING DASHBOARD
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EduPro Forecasting Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F5F7FA;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 40px;
        font-weight: 700;
        color: #1F4E79;
    }

    .dashboard-subtitle {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT FILES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


def file_path(filename):
    return os.path.join(
        BASE_DIR,
        filename
    )


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_CSV_FILES = [
    "EduPro_Next_Day_Forecast_Results.csv",
    "EduPro_Category_Forecast.csv",
    "EduPro_Enrollment_Model_Results.csv",
    "EduPro_Revenue_Forecast_Model_Results.csv",
    "EduPro_Enrollment_Forecast_Feature_Importance.csv",
    "EduPro_Revenue_Forecast_Feature_Importance.csv",
    "EduPro_Final_Forecasting_Dataset.csv"
]

REQUIRED_MODEL_FILES = [
    "EduPro_Next_Day_Enrollment_Model.pkl",
    "EduPro_Next_Day_Revenue_Model.pkl",
    "EduPro_Forecast_Features.pkl"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_existing_file(possible_names):

    for name in possible_names:

        path = file_path(name)

        if os.path.exists(path):

            return path

    return None


def safe_read_csv(filename):

    path = file_path(filename)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    return pd.read_csv(path)


def clean_column_names(df):

    df = df.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    return df


def get_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:

            return name

    return None


def numeric_series(df, column):

    if column is None:

        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():

    # --------------------------------------------------------
    # Forecast Results
    # --------------------------------------------------------

    forecast = safe_read_csv(
        "EduPro_Next_Day_Forecast_Results.csv"
    )

    # --------------------------------------------------------
    # Category Forecast
    # --------------------------------------------------------

    category_path = find_existing_file(
        [
            "EduPro_Category_Forecast.csv"
        ]
    )

    if category_path is not None:

        category = pd.read_csv(
            category_path
        )

    else:

        category = pd.DataFrame()

    # --------------------------------------------------------
    # Enrollment Results
    # --------------------------------------------------------

    enrollment_results = safe_read_csv(
        "EduPro_Enrollment_Model_Results.csv"
    )

    # --------------------------------------------------------
    # Revenue Results
    # --------------------------------------------------------

    revenue_results = safe_read_csv(
        "EduPro_Revenue_Forecast_Model_Results.csv"
    )

    # --------------------------------------------------------
    # Enrollment Feature Importance
    # --------------------------------------------------------

    enrollment_importance = safe_read_csv(
        "EduPro_Enrollment_Forecast_Feature_Importance.csv"
    )

    # --------------------------------------------------------
    # Revenue Feature Importance
    # --------------------------------------------------------

    revenue_importance = safe_read_csv(
        "EduPro_Revenue_Forecast_Feature_Importance.csv"
    )

    # --------------------------------------------------------
    # Final Dataset
    # --------------------------------------------------------

    final_data = safe_read_csv(
        "EduPro_Final_Forecasting_Dataset.csv"
    )

    return (
        clean_column_names(forecast),
        clean_column_names(category),
        clean_column_names(enrollment_results),
        clean_column_names(revenue_results),
        clean_column_names(enrollment_importance),
        clean_column_names(revenue_importance),
        clean_column_names(final_data)
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    enrollment_model = joblib.load(
        file_path(
            "EduPro_Next_Day_Enrollment_Model.pkl"
        )
    )

    revenue_model = joblib.load(
        file_path(
            "EduPro_Next_Day_Revenue_Model.pkl"
        )
    )

    forecast_features = joblib.load(
        file_path(
            "EduPro_Forecast_Features.pkl"
        )
    )

    return (
        enrollment_model,
        revenue_model,
        forecast_features
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        forecast,
        category,
        enrollment_results,
        revenue_results,
        enrollment_importance,
        revenue_importance,
        final_data
    ) = load_project_data()

except Exception as e:

    st.error(
        "❌ Error while loading project CSV files."
    )

    st.write(
        "Please make sure the CSV files are in the same "
        "folder as app.py."
    )

    st.exception(e)

    st.stop()


try:

    (
        enrollment_model,
        revenue_model,
        forecast_features
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Error while loading machine learning model files."
    )

    st.write(
        "Please make sure the three .pkl files are in "
        "the same folder as app.py."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DATE CONVERSION
# ============================================================

if "Date" in forecast.columns:

    forecast["Date"] = pd.to_datetime(
        forecast["Date"],
        errors="coerce"
    )


if "Date" in final_data.columns:

    final_data["Date"] = pd.to_datetime(
        final_data["Date"],
        errors="coerce"
    )


# ============================================================
# IDENTIFY MAIN COLUMNS
# ============================================================

actual_enrollment_col = get_column(
    forecast,
    [
        "Actual_Next_Day_Enrollment",
        "Actual_Enrollment",
        "Enrollment_Actual"
    ]
)

predicted_enrollment_col = get_column(
    forecast,
    [
        "Predicted_Next_Day_Enrollment",
        "Predicted_Enrollment",
        "Enrollment_Predicted"
    ]
)

actual_revenue_col = get_column(
    forecast,
    [
        "Actual_Next_Day_Revenue",
        "Actual_Revenue",
        "Revenue_Actual"
    ]
)

predicted_revenue_col = get_column(
    forecast,
    [
        "Predicted_Next_Day_Revenue",
        "Predicted_Revenue",
        "Revenue_Predicted"
    ]
)

course_category_col = get_column(
    forecast,
    [
        "CourseCategory",
        "Course_Category",
        "Category"
    ]
)

course_name_col = get_column(
    forecast,
    [
        "CourseName",
        "Course_Name"
    ]
)

course_id_col = get_column(
    forecast,
    [
        "CourseID",
        "Course_Id",
        "Course_ID"
    ]
)


# ============================================================
# CHECK FORECAST COLUMNS
# ============================================================

missing_forecast_columns = []

if actual_enrollment_col is None:

    missing_forecast_columns.append(
        "Actual enrollment"
    )

if predicted_enrollment_col is None:

    missing_forecast_columns.append(
        "Predicted enrollment"
    )

if actual_revenue_col is None:

    missing_forecast_columns.append(
        "Actual revenue"
    )

if predicted_revenue_col is None:

    missing_forecast_columns.append(
        "Predicted revenue"
    )

if course_category_col is None:

    missing_forecast_columns.append(
        "Course category"
    )


if missing_forecast_columns:

    st.error(
        "❌ Required forecast columns are missing."
    )

    st.write(
        "Detected columns:"
    )

    st.write(
        forecast.columns.tolist()
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    # 🎓 EduPro
    ### Forecasting Dashboard
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate to",
    [
        "🏠 Home",
        "📈 Demand Forecast",
        "💰 Revenue Forecast",
        "📚 Category Analysis",
        "🔍 Feature Importance",
        "🤖 Course Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Project**

    Predictive Modeling for Course Demand
    and Revenue Forecasting on EduPro

    **Technology**

    Python  
    Pandas  
    Scikit-learn  
    Plotly  
    Streamlit
    """
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="dashboard-title">'
        '🎓 EduPro Course Demand & Revenue Forecasting'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Predictive Analytics Dashboard for Course Demand, '
        'Revenue and Business Decision Support'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # KPI DATA
    # --------------------------------------------------------

    if course_id_col is not None:

        total_courses = (
            forecast[
                course_id_col
            ]
            .nunique()
        )

    else:

        total_courses = 0

    total_categories = (
        forecast[
            course_category_col
        ]
        .nunique()
    )

    total_records = len(
        forecast
    )

    avg_enrollment = (
        pd.to_numeric(
            forecast[
                actual_enrollment_col
            ],
            errors="coerce"
        )
        .mean()
    )

    actual_revenue = (
        pd.to_numeric(
            forecast[
                actual_revenue_col
            ],
            errors="coerce"
        )
        .sum()
    )

    predicted_revenue = (
        pd.to_numeric(
            forecast[
                predicted_revenue_col
            ],
            errors="coerce"
        )
        .sum()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "📚 Total Courses",
            f"{total_courses:,}"
        )

    with col2:

        st.metric(
            "🗂️ Categories",
            f"{total_categories:,}"
        )

    with col3:

        st.metric(
            "👥 Avg Enrollment",
            f"{avg_enrollment:.2f}"
        )

    with col4:

        st.metric(
            "💰 Actual Revenue",
            f"₹{actual_revenue:,.0f}"
        )

    with col5:

        st.metric(
            "🔮 Predicted Revenue",
            f"₹{predicted_revenue:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.header(
        "🎯 Project Objectives"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            ### 📈 Demand Forecasting

            Predict next-day course enrollment demand
            using historical transaction patterns and
            time-based forecasting features.
            """
        )

    with col2:

        st.success(
            """
            ### 💰 Revenue Forecasting

            Forecast next-day course revenue at course
            and category level.
            """
        )

    with col3:

        st.warning(
            """
            ### 📊 Decision Support

            Provide quantitative insights for course
            planning, pricing and instructor decisions.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # REVENUE BY CATEGORY
    # IMPORTANT:
    # DIRECTLY USE FORECAST DATA
    # --------------------------------------------------------

    st.header(
        "📊 Revenue by Course Category"
    )

    category_revenue = (
        forecast
        .groupby(
            course_category_col,
            as_index=False
        )
        .agg(
            Actual_Revenue=(
                actual_revenue_col,
                "sum"
            ),
            Predicted_Revenue=(
                predicted_revenue_col,
                "sum"
            )
        )
    )

    category_revenue = (
        category_revenue
        .sort_values(
            "Predicted_Revenue",
            ascending=False
        )
    )

    fig = px.bar(
        category_revenue,
        x=course_category_col,
        y=[
            "Actual_Revenue",
            "Predicted_Revenue"
        ],
        barmode="group",
        title="Actual vs Predicted Revenue by Category"
    )

    fig.update_layout(
        xaxis_title="Course Category",
        yaxis_title="Revenue",
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.header(
        "🤖 Machine Learning Models"
    )

    st.markdown(
        """
        The project evaluates multiple regression algorithms:

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor

        **Evaluation Metrics**

        - MAE — Mean Absolute Error
        - RMSE — Root Mean Squared Error
        - R² — Coefficient of Determination
        """
    )


# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "📈 Demand Forecast":

    st.title(
        "📈 Next-Day Enrollment Demand Forecast"
    )

    st.write(
        """
        Compare actual next-day enrollment with
        machine-learning predicted enrollment.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "🤖 Enrollment Model Performance"
    )

    enrollment_display = (
        enrollment_results.copy()
    )

    if "R2" in enrollment_display.columns:

        enrollment_display = (
            enrollment_display
            .sort_values(
                "R2",
                ascending=False
            )
        )

    st.dataframe(
        enrollment_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # R2 CHART
    # --------------------------------------------------------

    if (
        "Model" in enrollment_display.columns
        and
        "R2" in enrollment_display.columns
    ):

        fig = px.bar(
            enrollment_display,
            x="Model",
            y="R2",
            title="Enrollment Model Comparison — R²"
        )

        fig.update_layout(
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Enrollment"
    )

    plot_data = forecast[
        [
            actual_enrollment_col,
            predicted_enrollment_col
        ]
    ].copy()

    plot_data[
        actual_enrollment_col
    ] = pd.to_numeric(
        plot_data[
            actual_enrollment_col
        ],
        errors="coerce"
    )

    plot_data[
        predicted_enrollment_col
    ] = pd.to_numeric(
        plot_data[
            predicted_enrollment_col
        ],
        errors="coerce"
    )

    plot_data = plot_data.dropna()

    fig = px.scatter(
        plot_data,
        x=actual_enrollment_col,
        y=predicted_enrollment_col,
        title="Actual vs Predicted Next-Day Enrollment"
    )

    if len(plot_data) > 0:

        max_value = max(
            plot_data[
                actual_enrollment_col
            ].max(),

            plot_data[
                predicted_enrollment_col
            ].max()
        )

        min_value = min(
            plot_data[
                actual_enrollment_col
            ].min(),

            plot_data[
                predicted_enrollment_col
            ].min()
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    min_value,
                    max_value
                ],
                y=[
                    min_value,
                    max_value
                ],
                mode="lines",
                name="Perfect Prediction"
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # COURSE FILTER
    # --------------------------------------------------------

    st.subheader(
        "📚 Course-Level Forecast"
    )

    if course_category_col is not None:

        categories = sorted(
            forecast[
                course_category_col
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_category = st.selectbox(
            "Select Category",
            ["All"] + categories
        )

    else:

        selected_category = "All"

    filtered_forecast = forecast.copy()

    if selected_category != "All":

        filtered_forecast = (
            filtered_forecast[
                filtered_forecast[
                    course_category_col
                ].astype(str)
                == selected_category
            ]
        )

    display_columns = []

    for col in [
        "Date",
        course_id_col,
        course_name_col,
        course_category_col,
        actual_enrollment_col,
        predicted_enrollment_col
    ]:

        if (
            col is not None
            and
            col in filtered_forecast.columns
            and
            col not in display_columns
        ):

            display_columns.append(col)

    course_forecast_table = (
        filtered_forecast[
            display_columns
        ]
        .sort_values(
            predicted_enrollment_col,
            ascending=False
        )
    )

    st.dataframe(
        course_forecast_table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇️ Download Enrollment Forecast",
        data=forecast.to_csv(
            index=False
        ),
        file_name="EduPro_Enrollment_Forecast.csv",
        mime="text/csv"
    )


# ============================================================
# REVENUE FORECAST
# ============================================================

elif page == "💰 Revenue Forecast":

    st.title(
        "💰 Next-Day Revenue Forecast"
    )

    st.write(
        """
        Analyze actual and predicted next-day revenue
        using machine learning.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "🤖 Revenue Model Performance"
    )

    revenue_display = (
        revenue_results.copy()
    )

    if "R2" in revenue_display.columns:

        revenue_display = (
            revenue_display
            .sort_values(
                "R2",
                ascending=False
            )
        )

    st.dataframe(
        revenue_display,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # R2 CHART
    # --------------------------------------------------------

    if (
        "Model" in revenue_display.columns
        and
        "R2" in revenue_display.columns
    ):

        fig = px.bar(
            revenue_display,
            x="Model",
            y="R2",
            title="Revenue Model Comparison — R²"
        )

        fig.update_layout(
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED REVENUE
    # --------------------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Revenue"
    )

    plot_data = forecast[
        [
            actual_revenue_col,
            predicted_revenue_col
        ]
    ].copy()

    plot_data[
        actual_revenue_col
    ] = pd.to_numeric(
        plot_data[
            actual_revenue_col
        ],
        errors="coerce"
    )

    plot_data[
        predicted_revenue_col
    ] = pd.to_numeric(
        plot_data[
            predicted_revenue_col
        ],
        errors="coerce"
    )

    plot_data = plot_data.dropna()

    fig = px.scatter(
        plot_data,
        x=actual_revenue_col,
        y=predicted_revenue_col,
        title="Actual vs Predicted Next-Day Revenue"
    )

    if len(plot_data) > 0:

        max_value = max(
            plot_data[
                actual_revenue_col
            ].max(),

            plot_data[
                predicted_revenue_col
            ].max()
        )

        min_value = min(
            plot_data[
                actual_revenue_col
            ].min(),

            plot_data[
                predicted_revenue_col
            ].min()
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    min_value,
                    max_value
                ],
                y=[
                    min_value,
                    max_value
                ],
                mode="lines",
                name="Perfect Prediction"
            )
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TOP REVENUE COURSES
    # --------------------------------------------------------

    st.subheader(
        "💎 Top Courses by Predicted Revenue"
    )

    top_data = forecast.copy()

    top_data[
        predicted_revenue_col
    ] = pd.to_numeric(
        top_data[
            predicted_revenue_col
        ],
        errors="coerce"
    )

    top_data = (
        top_data
        .sort_values(
            predicted_revenue_col,
            ascending=False
        )
        .head(15)
    )

    if course_name_col is not None:

        fig = px.bar(
            top_data,
            x=predicted_revenue_col,
            y=course_name_col,
            orientation="h",
            title="Top 15 Courses by Predicted Revenue"
        )

        fig.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.dataframe(
            top_data,
            use_container_width=True
        )

    st.download_button(
        label="⬇️ Download Revenue Forecast",
        data=forecast.to_csv(
            index=False
        ),
        file_name="EduPro_Revenue_Forecast.csv",
        mime="text/csv"
    )


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

elif page == "📚 Category Analysis":

    st.title(
        "📚 Course Category Analysis"
    )

    st.write(
        """
        Compare actual and predicted enrollment and revenue
        across different course categories.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT USE category CSV FOR THESE COLUMNS
    # USE FORECAST DATA DIRECTLY
    # --------------------------------------------------------

    category_summary = (
        forecast
        .groupby(
            course_category_col,
            as_index=False
        )
        .agg(
            Actual_Enrollment=(
                actual_enrollment_col,
                "sum"
            ),
            Predicted_Enrollment=(
                predicted_enrollment_col,
                "sum"
            ),
            Actual_Revenue=(
                actual_revenue_col,
                "sum"
            ),
            Predicted_Revenue=(
                predicted_revenue_col,
                "sum"
            )
        )
    )

    category_summary = (
        category_summary
        .sort_values(
            "Predicted_Revenue",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    top_category = (
        category_summary.iloc[0][
            course_category_col
        ]
        if len(category_summary) > 0
        else "N/A"
    )

    top_revenue = (
        category_summary.iloc[0][
            "Predicted_Revenue"
        ]
        if len(category_summary) > 0
        else 0
    )

    k1, k2, k3 = st.columns(3)

    with k1:

        st.metric(
            "🗂️ Categories",
            len(category_summary)
        )

    with k2:

        st.metric(
            "🏆 Highest Predicted Revenue Category",
            str(top_category)
        )

    with k3:

        st.metric(
            "💰 Top Category Predicted Revenue",
            f"₹{top_revenue:,.0f}"
        )

    st.divider()

    # --------------------------------------------------------
    # ENROLLMENT CHART
    # --------------------------------------------------------

    st.subheader(
        "👥 Actual vs Predicted Enrollment"
    )

    fig = px.bar(
        category_summary,
        x=course_category_col,
        y=[
            "Actual_Enrollment",
            "Predicted_Enrollment"
        ],
        barmode="group",
        title="Enrollment by Course Category"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # REVENUE CHART
    # --------------------------------------------------------

    st.subheader(
        "💰 Actual vs Predicted Revenue"
    )

    fig = px.bar(
        category_summary,
        x=course_category_col,
        y=[
            "Actual_Revenue",
            "Predicted_Revenue"
        ],
        barmode="group",
        title="Revenue by Course Category"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "📊 Category Performance Table"
    )

    st.dataframe(
        category_summary,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇️ Download Category Analysis",
        data=category_summary.to_csv(
            index=False
        ),
        file_name="EduPro_Category_Analysis.csv",
        mime="text/csv"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.title(
        "🔍 Feature Importance Analysis"
    )

    st.write(
        """
        Feature importance indicates which variables were most
        influential in the Random Forest forecasting models.
        """
    )

    st.divider()

    target = st.radio(
        "Select Forecast Target",
        [
            "📈 Enrollment",
            "💰 Revenue"
        ],
        horizontal=True
    )

    if target == "📈 Enrollment":

        importance = (
            enrollment_importance
            .copy()
        )

        title = (
            "Top Drivers of Enrollment Forecast"
        )

    else:

        importance = (
            revenue_importance
            .copy()
        )

        title = (
            "Top Drivers of Revenue Forecast"
        )

    # --------------------------------------------------------
    # IDENTIFY FEATURE COLUMN
    # --------------------------------------------------------

    feature_column = None
    importance_column = None

    for col in importance.columns:

        lower_col = (
            str(col)
            .strip()
            .lower()
        )

        if lower_col in [
            "feature",
            "features",
            "variable",
            "feature_name"
        ]:

            feature_column = col

        if lower_col in [
            "importance",
            "feature_importance",
            "importance_score"
        ]:

            importance_column = col

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if feature_column is None:

        if len(importance.columns) >= 1:

            feature_column = (
                importance.columns[0]
            )

    if importance_column is None:

        if len(importance.columns) >= 2:

            importance_column = (
                importance.columns[1]
            )

    if (
        feature_column is None
        or
        importance_column is None
    ):

        st.error(
            "Could not identify feature importance columns."
        )

        st.write(
            importance.columns.tolist()
        )

        st.stop()

    # --------------------------------------------------------
    # NUMERIC IMPORTANCE
    # --------------------------------------------------------

    importance[
        importance_column
    ] = pd.to_numeric(
        importance[
            importance_column
        ],
        errors="coerce"
    )

    importance = (
        importance
        .dropna(
            subset=[
                importance_column
            ]
        )
        .sort_values(
            importance_column,
            ascending=False
        )
    )

    # --------------------------------------------------------
    # TOP FEATURES
    # --------------------------------------------------------

    top_features = (
        importance
        .head(15)
        .sort_values(
            importance_column
        )
    )

    fig = px.bar(
        top_features,
        x=importance_column,
        y=feature_column,
        orientation="h",
        title=title
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "📊 Complete Feature Importance"
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇️ Download Feature Importance",
        data=importance.to_csv(
            index=False
        ),
        file_name="EduPro_Feature_Importance.csv",
        mime="text/csv"
    )


# ============================================================
# COURSE PREDICTION
# ============================================================

elif page == "🤖 Course Prediction":

    st.title(
        "🤖 Interactive Course Prediction"
    )

    st.write(
        """
        Select a course and modify important course/instructor
        parameters to generate a next-day forecast.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CHECK FINAL DATA
    # --------------------------------------------------------

    if (
        "CourseID" not in final_data.columns
        or
        "CourseName" not in final_data.columns
    ):

        st.error(
            "Course information is not available in "
            "EduPro_Final_Forecasting_Dataset.csv."
        )

        st.stop()

    # --------------------------------------------------------
    # COURSE LIST
    # --------------------------------------------------------

    course_list = (
        final_data[
            [
                "CourseID",
                "CourseName"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "CourseName"
        )
    )

    if len(course_list) == 0:

        st.error(
            "No courses available."
        )

        st.stop()

    selected_course_name = st.selectbox(
        "📚 Select Course",
        course_list[
            "CourseName"
        ].astype(str).tolist()
    )

    selected_course_id = course_list.loc[
        course_list[
            "CourseName"
        ].astype(str)
        == selected_course_name,
        "CourseID"
    ].iloc[0]

    # --------------------------------------------------------
    # COURSE DATA
    # --------------------------------------------------------

    course_rows = final_data[
        final_data[
            "CourseID"
        ] == selected_course_id
    ].copy()

    if len(course_rows) == 0:

        st.error(
            "No data found for selected course."
        )

        st.stop()

    if "Date" in course_rows.columns:

        course_rows = (
            course_rows
            .sort_values(
                "Date"
            )
        )

    base_row = (
        course_rows
        .iloc[-1]
        .copy()
    )

    # --------------------------------------------------------
    # SAFE DEFAULT FUNCTION
    # --------------------------------------------------------

    def safe_float(
        row,
        column,
        default
    ):

        if column not in row.index:

            return float(default)

        value = pd.to_numeric(
            row[column],
            errors="coerce"
        )

        if pd.isna(value):

            return float(default)

        return float(value)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        course_price = st.number_input(
            "💵 Course Price",
            min_value=0.0,
            value=safe_float(
                base_row,
                "CoursePrice",
                499
            ),
            step=50.0
        )

        course_duration = st.number_input(
            "⏱️ Course Duration",
            min_value=1.0,
            value=safe_float(
                base_row,
                "CourseDuration",
                10
            ),
            step=1.0
        )

        course_rating = st.number_input(
            "⭐ Course Rating",
            min_value=0.0,
            max_value=5.0,
            value=safe_float(
                base_row,
                "CourseRating",
                4.0
            ),
            step=0.1
        )

    with col2:

        years_experience = st.number_input(
            "👨‍🏫 Teacher Experience",
            min_value=0.0,
            value=safe_float(
                base_row,
                "YearsOfExperience",
                5
            ),
            step=1.0
        )

        teacher_rating = st.number_input(
            "⭐ Teacher Rating",
            min_value=0.0,
            max_value=5.0,
            value=safe_float(
                base_row,
                "TeacherRating",
                4.0
            ),
            step=0.1
        )

    st.divider()

    # --------------------------------------------------------
    # COURSE INFO
    # --------------------------------------------------------

    st.subheader(
        "📋 Selected Course Information"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "Course",
            str(
                base_row.get(
                    "CourseName",
                    selected_course_name
                )
            )
        )

    with info2:

        st.metric(
            "Category",
            str(
                base_row.get(
                    "CourseCategory",
                    "Unknown"
                )
            )
        )

    with info3:

        st.metric(
            "Level",
            str(
                base_row.get(
                    "CourseLevel",
                    "Unknown"
                )
            )
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🚀 Predict Next-Day Demand & Revenue",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        try:

            # ------------------------------------------------
            # CREATE INPUT ROW
            # ------------------------------------------------

            input_data = {}

            for feature in forecast_features:

                if feature in base_row.index:

                    input_data[
                        feature
                    ] = base_row[
                        feature
                    ]

                else:

                    input_data[
                        feature
                    ] = 0

            input_df = pd.DataFrame(
                [input_data]
            )

            # ------------------------------------------------
            # USER INPUT OVERRIDES
            # ------------------------------------------------

            override_values = {

                "CoursePrice":
                    course_price,

                "CourseDuration":
                    course_duration,

                "CourseRating":
                    course_rating,

                "YearsOfExperience":
                    years_experience,

                "TeacherRating":
                    teacher_rating
            }

            for col, value in override_values.items():

                if col in input_df.columns:

                    input_df[
                        col
                    ] = value

            # ------------------------------------------------
            # FORECAST DATE
            # ------------------------------------------------

            if (
                "Date" in final_data.columns
                and
                final_data["Date"].notna().any()
            ):

                last_date = (
                    final_data[
                        "Date"
                    ]
                    .max()
                )

                prediction_date = (
                    last_date
                    +
                    pd.Timedelta(
                        days=1
                    )
                )

            else:

                prediction_date = (
                    pd.Timestamp.today()
                )

            # ------------------------------------------------
            # CALENDAR FEATURES
            # ------------------------------------------------

            if "Year" in input_df.columns:

                input_df[
                    "Year"
                ] = prediction_date.year

            if "Month" in input_df.columns:

                input_df[
                    "Month"
                ] = prediction_date.month

            if "Quarter" in input_df.columns:

                input_df[
                    "Quarter"
                ] = prediction_date.quarter

            if "Day" in input_df.columns:

                input_df[
                    "Day"
                ] = prediction_date.day

            if "DayOfWeek" in input_df.columns:

                input_df[
                    "DayOfWeek"
                ] = prediction_date.dayofweek

            if "WeekOfYear" in input_df.columns:

                input_df[
                    "WeekOfYear"
                ] = prediction_date.isocalendar().week

            if "IsWeekend" in input_df.columns:

                input_df[
                    "IsWeekend"
                ] = int(
                    prediction_date.dayofweek >= 5
                )

            if "DayOfYear" in input_df.columns:

                input_df[
                    "DayOfYear"
                ] = prediction_date.dayofyear

            # ------------------------------------------------
            # CYCLICAL FEATURES
            # ------------------------------------------------

            if "Month_Sin" in input_df.columns:

                input_df[
                    "Month_Sin"
                ] = np.sin(
                    2 * np.pi
                    *
                    prediction_date.month
                    / 12
                )

            if "Month_Cos" in input_df.columns:

                input_df[
                    "Month_Cos"
                ] = np.cos(
                    2 * np.pi
                    *
                    prediction_date.month
                    / 12
                )

            if "DayOfWeek_Sin" in input_df.columns:

                input_df[
                    "DayOfWeek_Sin"
                ] = np.sin(
                    2 * np.pi
                    *
                    prediction_date.dayofweek
                    / 7
                )

            if "DayOfWeek_Cos" in input_df.columns:

                input_df[
                    "DayOfWeek_Cos"
                ] = np.cos(
                    2 * np.pi
                    *
                    prediction_date.dayofweek
                    / 7
                )

            # ------------------------------------------------
            # ENSURE EXACT FEATURE ORDER
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in forecast_features
                if feature not in input_df.columns
            ]

            if missing_features:

                for feature in missing_features:

                    input_df[
                        feature
                    ] = 0

            input_df = input_df[
                forecast_features
            ]

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            enrollment_prediction = (
                enrollment_model
                .predict(
                    input_df
                )[0]
            )

            revenue_prediction = (
                revenue_model
                .predict(
                    input_df
                )[0]
            )

            # ------------------------------------------------
            # CLEAN PREDICTIONS
            # ------------------------------------------------

            enrollment_prediction = max(
                0,
                float(
                    enrollment_prediction
                )
            )

            revenue_prediction = max(
                0,
                float(
                    revenue_prediction
                )
            )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.success(
                "✅ Prediction completed successfully!"
            )

            st.subheader(
                "🔮 Prediction Results"
            )

            result1, result2 = st.columns(2)

            with result1:

                st.metric(
                    "👥 Predicted Next-Day Enrollment",
                    f"{enrollment_prediction:.2f}"
                )

            with result2:

                st.metric(
                    "💰 Predicted Next-Day Revenue",
                    f"₹{revenue_prediction:,.2f}"
                )

            st.divider()

            st.subheader(
                "📅 Forecast Details"
            )

            detail1, detail2 = st.columns(2)

            with detail1:

                st.write(
                    f"**Course:** "
                    f"{selected_course_name}"
                )

                st.write(
                    f"**Prediction Date:** "
                    f"{prediction_date.date()}"
                )

                st.write(
                    f"**Course Price:** "
                    f"₹{course_price:,.2f}"
                )

            with detail2:

                st.write(
                    f"**Course Duration:** "
                    f"{course_duration}"
                )

                st.write(
                    f"**Teacher Experience:** "
                    f"{years_experience:.0f} years"
                )

                st.write(
                    f"**Teacher Rating:** "
                    f"{teacher_rating:.1f}/5"
                )

        except Exception as e:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.write(
                """
                The dashboard itself is working, but the
                interactive prediction encountered a problem.
                """
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777;">

    <b>EduPro Predictive Analytics Dashboard</b>
    <br>
    Course Demand & Revenue Forecasting using Machine Learning
    <br>
    Built with Python • Pandas • Scikit-learn • Plotly • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)