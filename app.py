"""
Jobs & Skills Data Science Pipeline - Streamlit App
Complete ML pipeline with visualizations, model comparison, SHAP analysis, and feature selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Note: TensorFlow is NOT imported here. MLP results are loaded from saved artifacts
# (test_results.json, mlp_history.json, y_pred_test_mlp.npy) to avoid compatibility issues.

# ──────────────────────────────────────────────────────────
# CONFIG & DATA LOADING
# ──────────────────────────────────────────────────────────
import os
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Jobs & Skills ML Pipeline", layout="wide", page_icon="📊")

# ──────────────────────────────────────────────────────────
# CUSTOM CSS THEME
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Main container breathing room ── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2838 0%, #2C3E50 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ECF0F1 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f8f9fc;
    padding: 6px 12px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 20px;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.95rem;
    color: #555;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1B2838 !important;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-radius: 10px;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #636EFA !important;
    height: 3px;
    border-radius: 3px;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf1;
    border-top: 4px solid #636EFA;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,110,250,0.12);
}
div[data-testid="stMetric"] label {
    color: #6B7280 !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1B2838 !important;
    font-weight: 700;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #636EFA !important;
    font-weight: 500;
}

/* ── Dataframe styling ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #e8ecf1;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

/* ── Expander styling ── */
div[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #2C3E50;
}

/* ── Section headers ── */
h3 {
    color: #1B2838 !important;
    font-weight: 700 !important;
    border-left: 4px solid #636EFA;
    padding-left: 12px;
    margin-top: 1.5rem !important;
}

/* ── Styled dividers ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, #636EFA 0%, #e8ecf1 50%, transparent 100%);
    margin: 2rem 0;
}

/* ── Info/callout boxes ── */
.callout-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #f8f9fc 100%);
    border-left: 5px solid #636EFA;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 16px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.callout-box-green {
    background: linear-gradient(135deg, #f0faf5 0%, #f8fcf9 100%);
    border-left: 5px solid #00CC96;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 16px 0;
}
.callout-box-purple {
    background: linear-gradient(135deg, #f5f0ff 0%, #faf8fc 100%);
    border-left: 5px solid #AB63FA;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 16px 0;
}

/* ── Prediction result card ── */
.prediction-card {
    background: linear-gradient(135deg, #636EFA 0%, #4854d4 100%);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(99,110,250,0.25);
    margin: 16px 0;
}
.prediction-card .pred-value {
    font-size: 3rem;
    font-weight: 800;
    margin: 8px 0;
}
.prediction-card .pred-label {
    font-size: 1rem;
    opacity: 0.9;
    font-weight: 500;
}
.prediction-card .pred-subtitle {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 8px;
}

/* ── Feature input panel ── */
.input-panel {
    background: #f8f9fc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    margin: 12px 0;
}

/* ── Selectbox & slider polish ── */
div[data-baseweb="select"] {
    border-radius: 8px;
}

/* ── Plotly chart containers ── */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Badge / pill styling ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px 4px;
}
.badge-blue { background: #EEF2FF; color: #636EFA; }
.badge-green { background: #ECFDF5; color: #059669; }
.badge-purple { background: #F5F3FF; color: #7C3AED; }
.badge-orange { background: #FFF7ED; color: #D97706; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_parquet(f"{ARTIFACT_DIR}/processed_data.parquet")
    X = np.load(f"{ARTIFACT_DIR}/X.npy")
    X_scaled = np.load(f"{ARTIFACT_DIR}/X_scaled.npy")
    y = np.load(f"{ARTIFACT_DIR}/y.npy")
    with open(f"{ARTIFACT_DIR}/config.json") as f:
        config = json.load(f)
    with open(f"{ARTIFACT_DIR}/cv_results.json") as f:
        cv_results = json.load(f)
    return df, X, X_scaled, y, config, cv_results

@st.cache_resource
def load_models():
    model_names = ['linear_regression', 'lasso', 'ridge', 'cart', 'random_forest', 'lightgbm']
    models = {}
    display_names = {
        'linear_regression': 'Linear Regression',
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'cart': 'CART',
        'random_forest': 'Random Forest',
        'lightgbm': 'LightGBM'
    }
    for mn in model_names:
        models[display_names[mn]] = joblib.load(f"{ARTIFACT_DIR}/model_{mn}.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.joblib")
    label_encoders = joblib.load(f"{ARTIFACT_DIR}/label_encoders.joblib")
    return models, scaler, label_encoders

@st.cache_data
def load_shap_data():
    X_shap = np.load(f"{ARTIFACT_DIR}/X_shap.npy")
    X_shap_scaled = np.load(f"{ARTIFACT_DIR}/X_shap_scaled.npy")
    shap_dict = {}
    for name in ['linear_regression', 'lasso', 'ridge', 'cart', 'random_forest', 'lightgbm']:
        shap_dict[name] = np.load(f"{ARTIFACT_DIR}/shap_{name}.npy")
    return X_shap, X_shap_scaled, shap_dict

@st.cache_data
def load_part2_data():
    """Load Part 2 artifacts: train/test split, GridSearchCV results, MLP history."""
    try:
        y_test = np.load(f"{ARTIFACT_DIR}/y_test.npy")
        with open(f"{ARTIFACT_DIR}/test_results.json") as f:
            test_results = json.load(f)
        grid_params = {}
        for key in ['cart', 'rf', 'lgb']:
            try:
                with open(f"{ARTIFACT_DIR}/grid_{key}_best_params.json") as f:
                    grid_params[key] = json.load(f)
            except FileNotFoundError:
                grid_params[key] = {}
        predictions = {}
        for name in test_results.keys():
            safe = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            try:
                predictions[name] = np.load(f"{ARTIFACT_DIR}/y_pred_test_{safe}.npy")
            except FileNotFoundError:
                pass
        try:
            with open(f"{ARTIFACT_DIR}/mlp_history.json") as f:
                mlp_history = json.load(f)
        except FileNotFoundError:
            mlp_history = None
        return y_test, test_results, grid_params, predictions, mlp_history
    except FileNotFoundError:
        return None, None, None, None, None

df, X, X_scaled, y, config, cv_results = load_data()
models, scaler, label_encoders = load_models()
X_shap, X_shap_scaled, shap_values_dict = load_shap_data()
feature_cols = config['feature_cols']
y_test_part2, test_results_p2, grid_params, predictions_p2, mlp_history = load_part2_data()

DISPLAY_TO_SAFE = {
    'Linear Regression': 'linear_regression',
    'Lasso': 'lasso',
    'Ridge': 'ridge',
    'CART': 'cart',
    'Random Forest': 'random_forest',
    'LightGBM': 'lightgbm'
}

# ──────────────────────────────────────────────────────────
# UNIFIED PLOTLY THEME
# ──────────────────────────────────────────────────────────
PLOT_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

plotly_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13, color="#2C3E50"),
        title=dict(font=dict(size=18, color="#1B2838"), x=0.0, xanchor='left'),
        paper_bgcolor='white',
        plot_bgcolor='#FAFBFD',
        colorway=PLOT_COLORS,
        xaxis=dict(gridcolor='#E8ECF1', linecolor='#D1D5DB', linewidth=1, zeroline=False),
        yaxis=dict(gridcolor='#E8ECF1', linecolor='#D1D5DB', linewidth=1, zeroline=False),
        margin=dict(l=60, r=30, t=60, b=50),
        hoverlabel=dict(bgcolor='white', bordercolor='#636EFA', font_size=13),
        legend=dict(bgcolor='rgba(255,255,255,0.85)', bordercolor='#E8ECF1', borderwidth=1, font=dict(size=12)),
    )
)
import plotly.io as pio
pio.templates['custom'] = plotly_template
pio.templates.default = 'plotly+custom'

# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div style="font-size: 2.5rem; margin-bottom: 4px;">📊</div>
    <div style="font-size: 1.3rem; font-weight: 700; color: #ECF0F1;">Jobs & Skills</div>
    <div style="font-size: 0.85rem; color: #95A5A6; font-weight: 400;">ML Pipeline Dashboard</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin: 8px 0;">
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #95A5A6; margin-bottom: 10px; font-weight: 600;">Dataset Overview</div>
    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
        <span style="color: #BDC3C7;">Samples</span>
        <span style="color: #ECF0F1; font-weight: 700;">""" + f"{len(df):,}" + """</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
        <span style="color: #BDC3C7;">Features</span>
        <span style="color: #ECF0F1; font-weight: 700;">""" + str(len(feature_cols)) + """</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
        <span style="color: #BDC3C7;">Target</span>
        <span style="color: #ECF0F1; font-weight: 700;">num_skills</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin: 8px 0;">
        <span style="color: #BDC3C7;">Models</span>
        <span style="color: #ECF0F1; font-weight: 700;">7 (incl. MLP)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background: rgba(99,110,250,0.15); border-radius: 12px; padding: 16px; margin: 8px 0;">
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #95A5A6; margin-bottom: 8px; font-weight: 600;">Best Model</div>
    <div style="color: #636EFA; font-weight: 700; font-size: 1.1rem;">LightGBM</div>
    <div style="color: #BDC3C7; font-size: 0.85rem; margin-top: 4px;">RMSE: 10.48 &nbsp;|&nbsp; R&sup2;: 0.205</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 8px 0; color: #7F8C8D; font-size: 0.75rem;">
    MSIS 522B &bull; LinkedIn Job Postings<br>
    Predictive Analytics Project
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# HORIZONTAL TABS
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Interactive Prediction"
])

# ──────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE SUMMARY
# ──────────────────────────────────────────────────────────
with tab1:
    st.title("📋 Executive Summary")

    st.markdown("""
    <div class="callout-box">
        <h4 style="margin-top:0; color:#1B2838;">Dataset & Prediction Task</h4>
        This project analyzes <strong>LinkedIn job postings</strong> sourced from Kaggle, consisting of two datasets:
        <code>linkedin_job_postings.csv</code> (job metadata including title, company, location, seniority level, and
        work type) and <code>job_skills.csv</code> (individual skills listed per posting). After merging on <code>job_link</code>
        and feature engineering, the final dataset contains <strong>80,000 job postings</strong> with <strong>21 features</strong>
        across the United States, United Kingdom, Canada, and Australia. The features include 10 numerical
        variables (title length, company name length, average skill name length, day of week, etc.) and
        11 binary flags indicating the presence of common skills (communication, management, data, leadership,
        Python, Excel, sales, marketing, project management, customer). The <strong>prediction target</strong> is
        <code>num_skills</code> — the count of distinct skills listed per job posting — making this a <strong>regression task</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box-green">
        <h4 style="margin-top:0; color:#059669;">Why This Matters</h4>
        Understanding what drives skill requirements in job postings has direct value across the labor market.
        <strong>Job seekers</strong> can better prepare by knowing which role attributes correlate with broader skill
        demands — for example, remote positions require ~24 skills on average versus ~20 for onsite roles.
        <strong>Recruiters and hiring managers</strong> can set more realistic expectations when drafting postings, avoiding
        the common pitfall of listing too many or too few skills. <strong>Job platform designers</strong> (LinkedIn, Indeed)
        could use such a model to auto-suggest skill tags for new postings, improving discoverability.
        <strong>Workforce analysts</strong> gain insight into how skill demands vary by geography, seniority, and industry,
        informing education and training policy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box-purple">
        <h4 style="margin-top:0; color:#7C3AED;">Approach & Key Findings</h4>
        We trained <strong>seven models</strong> — Linear Regression, Lasso, Ridge, CART (decision tree), Random Forest,
        LightGBM, and a Multi-Layer Perceptron (MLP) neural network — using a 70/30 train/test split.
        Tree-based models (CART, RF, LightGBM) were tuned via GridSearchCV with 5-fold cross-validation.
        <strong>LightGBM achieved the best test-set performance</strong> (RMSE = 10.48, R&sup2; = 0.205), followed by the
        MLP (RMSE = 10.63) and Random Forest (RMSE = 10.70). Linear models served as baselines
        (RMSE &asymp; 10.91). SHAP analysis revealed that the binary skill-presence flags (<code>has_management</code>,
        <code>has_data</code>, <code>has_communication</code>) are the strongest predictors — postings that mention broad,
        cross-functional skills tend to list more total skills. Geographic and temporal features had
        weak effects. These insights suggest that <strong>role complexity, not technical specialization</strong>,
        is the primary driver of skill breadth in job postings.
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    best_model = max(cv_results.items(), key=lambda x: x[1]['r2_mean'])
    worst_model = min(cv_results.items(), key=lambda x: x[1]['r2_mean'])

    col1.metric("Best Model (CV R²)", best_model[0], f"{best_model[1]['r2_mean']:.4f}")
    col2.metric("Best CV RMSE", best_model[0], f"{best_model[1]['rmse_mean']:.2f}")
    col3.metric("Avg Skills / Job", f"{y.mean():.1f}", f"σ = {y.std():.1f}")
    col4.metric("Total Features", f"{len(feature_cols)}", f"{len(df):,} samples")

    st.markdown("---")
    st.markdown("#### Model Performance Summary (5-Fold Cross-Validation)")

    results_df = pd.DataFrame(cv_results).T
    results_df.index.name = 'Model'
    results_df = results_df.reset_index()
    results_df.columns = ['Model', 'CV RMSE (mean)', 'RMSE (std)', 'CV MAE (mean)', 'MAE (std)',
                          'CV R² (mean)', 'R² (std)', 'Train RMSE', 'Train R²']

    # Highlight best
    st.dataframe(
        results_df.style.highlight_max(subset=['CV R² (mean)', 'Train R²'], color='#c6efce')
                        .highlight_min(subset=['CV RMSE (mean)', 'CV MAE (mean)'], color='#c6efce')
                        .format({
                            'CV RMSE (mean)': '{:.4f}', 'RMSE (std)': '{:.4f}',
                            'CV MAE (mean)': '{:.4f}', 'MAE (std)': '{:.4f}',
                            'CV R² (mean)': '{:.4f}', 'R² (std)': '{:.4f}',
                            'Train RMSE': '{:.4f}', 'Train R²': '{:.4f}'
                        }),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Key Findings")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="callout-box" style="border-left-color: #EF553B;">
            <h5 style="margin-top:0; color:#EF553B;">Model Insights</h5>
            <ul style="margin-bottom:0;">
                <li><strong>LightGBM</strong> achieves the best cross-validated R&sup2; and lowest RMSE, confirming gradient boosting's
                  ability to capture non-linear feature interactions</li>
                <li><strong>Random Forest</strong> comes in second, with ensemble averaging reducing variance</li>
                <li><strong>Linear models</strong> (OLS, Ridge, Lasso) perform comparably to each other; Ridge and OLS are nearly
                  identical, while Lasso slightly penalizes less important features</li>
                <li><strong>CART</strong> (single decision tree) has the weakest performance due to high variance and overfitting risk</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="callout-box" style="border-left-color: #00CC96;">
            <h5 style="margin-top:0; color:#059669;">Feature Insights (from SHAP)</h5>
            <ul style="margin-bottom:0;">
                <li><strong>Average skill name length</strong> is the strongest predictor — jobs with longer/more specific skill
                  names tend to require more skills overall</li>
                <li><strong>Job title length</strong> and <strong>word count</strong> contribute meaningfully, suggesting more complex roles
                  (with longer titles) demand broader skill sets</li>
                <li><strong>Skill flags</strong> (has_communication, has_management, etc.) act as strong signals for skill-rich
                  postings</li>
                <li><strong>Geographic and temporal features</strong> have moderate effects, with some states and posting months
                  showing subtle patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📘 Methodology (click to expand)", expanded=False):
        st.markdown("""
        1. **Data Merging**: LinkedIn job postings joined with job skills on `job_link`
        2. **Feature Engineering**: 21 features derived from categorical encoding, text metrics, date extraction,
           and binary skill presence flags
        3. **Modeling**: Seven algorithms trained — Linear Regression, Lasso, Ridge,
           CART, Random Forest, LightGBM, MLP — with 70/30 train/test split
        4. **Hyperparameter Tuning**: GridSearchCV (5-fold CV) for CART, Random Forest, LightGBM;
           Early Stopping for MLP
        5. **Explainability**: SHAP (SHapley Additive exPlanations) values computed for all six tree/linear models
           using TreeExplainer and LinearExplainer
        6. **Interactive Prediction**: Users can set feature values via sliders/dropdowns and see real-time
           predictions with SHAP waterfall explanations
        """)


# ──────────────────────────────────────────────────────────
# TAB 2: DATA VISUALIZATION
# ──────────────────────────────────────────────────────────
with tab2:
    st.title("📊 Descriptive Analytics")

    st.markdown("### Target Variable Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='num_skills', nbins=50, color_discrete_sequence=['#636EFA'],
                           title='Distribution of Number of Skills per Job')
        fig.add_vline(x=df['num_skills'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['num_skills'].mean():.1f}")
        fig.add_vline(x=df['num_skills'].median(), line_dash="dot", line_color="green",
                      annotation_text=f"Median: {df['num_skills'].median():.0f}")
        fig.update_layout(xaxis_title="Number of Skills", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y='num_skills', color_discrete_sequence=['#EF553B'],
                     title='Box Plot of Skills Count')
        fig.update_layout(yaxis_title="Number of Skills", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Skills by Job Level & Type")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.violin(df, x='job_level', y='num_skills', color='job_level', box=True,
                        title='Skills Distribution by Job Level',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(df, x='job_type', y='num_skills', color='job_type', box=True,
                        title='Skills Distribution by Job Type',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Geographic Analysis")
    col1, col2 = st.columns(2)

    with col1:
        country_stats = df.groupby('search_country')['num_skills'].agg(['mean', 'median', 'count']).reset_index()
        country_stats.columns = ['Country', 'Mean Skills', 'Median Skills', 'Job Count']
        fig = px.bar(country_stats, x='Country', y='Mean Skills', color='Country',
                     text='Job Count', title='Average Skills by Country',
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(texttemplate='n=%{text:,}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_states = df.groupby('state')['num_skills'].agg(['mean', 'count']).reset_index()
        top_states.columns = ['State', 'Mean Skills', 'Count']
        top_states = top_states[top_states['Count'] >= 100].nlargest(20, 'Mean Skills')
        fig = px.bar(top_states, x='Mean Skills', y='State', orientation='h',
                     color='Mean Skills', title='Top 20 States by Avg Skills (min 100 jobs)',
                     color_continuous_scale='Viridis')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Common Skill Prevalence")
    common_skills = config['common_skills']
    skill_flags = [f'has_{s.lower().replace(" ", "_")}' for s in common_skills]
    skill_pct = df[skill_flags].mean() * 100
    skill_pct.index = common_skills

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=skill_pct.values, y=skill_pct.index, orientation='h',
                     color=skill_pct.values, color_continuous_scale='RdYlGn',
                     title='Prevalence of Common Skills (%)',
                     labels={'x': 'Percentage of Jobs', 'y': 'Skill'})
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Mean skills when skill is present vs absent
        comparison = []
        for skill, flag in zip(common_skills, skill_flags):
            mean_with = df[df[flag] == 1]['num_skills'].mean()
            mean_without = df[df[flag] == 0]['num_skills'].mean()
            comparison.append({'Skill': skill, 'With Skill': mean_with, 'Without Skill': mean_without})
        comp_df = pd.DataFrame(comparison).melt(id_vars='Skill', var_name='Presence', value_name='Avg Skills')
        fig = px.bar(comp_df, x='Skill', y='Avg Skills', color='Presence', barmode='group',
                     title='Avg Skills Count: With vs Without Common Skill',
                     color_discrete_sequence=['#00CC96', '#EF553B'])
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Correlations")
    corr_df = df[feature_cols + ['num_skills']].corr()
    fig = px.imshow(corr_df, text_auto='.2f', color_continuous_scale='RdBu_r',
                    title='Feature Correlation Heatmap', aspect='auto',
                    zmin=-1, zmax=1)
    fig.update_layout(height=700, width=900)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Title Length vs Skills")
    col1, col2 = st.columns(2)
    with col1:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(sample, x='title_length', y='num_skills', color='job_level',
                         opacity=0.3, title='Title Length vs Number of Skills',
                         color_discrete_sequence=px.colors.qualitative.Set1)
        m, b = np.polyfit(sample['title_length'], sample['num_skills'], 1)
        x_range = np.array([sample['title_length'].min(), sample['title_length'].max()])
        fig.add_trace(go.Scatter(x=x_range, y=m * x_range + b, mode='lines',
                                 line=dict(color='black', dash='dash'), name='OLS Trend'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(sample, x='avg_skill_length', y='num_skills', color='search_country',
                         opacity=0.3, title='Avg Skill Name Length vs Number of Skills',
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        m, b = np.polyfit(sample['avg_skill_length'], sample['num_skills'], 1)
        x_range = np.array([sample['avg_skill_length'].min(), sample['avg_skill_length'].max()])
        fig.add_trace(go.Scatter(x=x_range, y=m * x_range + b, mode='lines',
                                 line=dict(color='black', dash='dash'), name='OLS Trend'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Temporal Patterns")
    col1, col2 = st.columns(2)
    with col1:
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_stats = df.groupby('day_of_week')['num_skills'].agg(['mean', 'count']).reset_index()
        dow_stats['Day'] = dow_stats['day_of_week'].map(lambda x: dow_names[int(x)] if x < 7 else 'Unk')
        fig = px.bar(dow_stats, x='Day', y='mean', text='count',
                     title='Avg Skills by Day of Week',
                     color_discrete_sequence=['#AB63FA'])
        fig.update_traces(texttemplate='n=%{text:,}', textposition='outside')
        fig.update_layout(height=400, yaxis_title='Avg Skills')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        month_stats = df.groupby('month')['num_skills'].agg(['mean', 'count']).reset_index()
        fig = px.line(month_stats, x='month', y='mean', markers=True,
                      title='Avg Skills by Month',
                      color_discrete_sequence=['#FFA15A'])
        fig.update_layout(height=400, xaxis_title='Month', yaxis_title='Avg Skills')
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab3:
    st.title("🤖 Model Performance")

    if test_results_p2 is None or grid_params is None:
        st.warning("⚠️ Part 2 artifacts not found. Run `python train_pipeline.py` first.")
    else:
        st.markdown("""
        All models trained on the **70% training set** (56,000 samples) and evaluated on the
        **30% held-out test set** (24,000 samples) with `random_state=42`. Tree-based models
        tuned via `GridSearchCV` with 5-fold CV; MLP trained with early stopping.
        """)

        # ── 2.2 BASELINE ──
        st.markdown("---")
        st.markdown("### 2.2 Linear Regression Baseline")
        st.markdown("""
        A **Linear Regression** model serves as the baseline. We also include **Lasso** (L1) and
        **Ridge** (L2) variants. All three use standardized features.
        """)
        baseline_names = ['Linear Regression', 'Lasso', 'Ridge']
        baseline_rows = []
        for bn in baseline_names:
            res = test_results_p2.get(bn, {})
            if res:
                baseline_rows.append({'Model': bn, 'Test RMSE': res['test_rmse'],
                                      'Test MAE': res['test_mae'], 'Test R²': res['test_r2']})
        if baseline_rows:
            baseline_df = pd.DataFrame(baseline_rows)
            st.dataframe(
                baseline_df.style
                    .highlight_min(subset=['Test RMSE', 'Test MAE'], color='#c6efce')
                    .highlight_max(subset=['Test R²'], color='#c6efce')
                    .format({'Test RMSE': '{:.4f}', 'Test MAE': '{:.4f}', 'Test R²': '{:.4f}'}),
                use_container_width=True
            )
            best_bl = min(baseline_rows, key=lambda x: x['Test RMSE'])
            st.markdown(f"**Baseline benchmark:** RMSE = **{best_bl['Test RMSE']:.4f}**, "
                        f"R² = **{best_bl['Test R²']:.4f}**. All subsequent models should beat this.")

        # ── 2.3 CART ──
        st.markdown("---")
        st.markdown("### 2.3 Decision Tree (CART) — GridSearchCV")
        cart_bp = grid_params.get('cart', {})
        st.markdown(f"**Best hyperparameters:** `max_depth={cart_bp.get('max_depth')}`, "
                    f"`min_samples_leaf={cart_bp.get('min_samples_leaf')}`")
        st.markdown("**Param grid:** max_depth=[3,5,7,10], min_samples_leaf=[5,10,20,50] → 16 combos × 5 folds")

        try:
            cart_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_cart_cv_results.json")
            pivot = cart_cv.pivot_table(values='mean_test_score',
                        index='param_min_samples_leaf', columns='param_max_depth')
            pivot_rmse = np.sqrt(-pivot)
            fig = px.imshow(pivot_rmse, text_auto='.3f', color_continuous_scale='RdYlGn_r',
                            title='CART GridSearchCV: Mean CV RMSE',
                            labels=dict(x='max_depth', y='min_samples_leaf', color='RMSE'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        cart_res = test_results_p2.get('CART (Tuned)', {})
        if cart_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{cart_res['test_rmse']:.4f}")
            c2.metric("Test MAE", f"{cart_res['test_mae']:.4f}")
            c3.metric("Test R²", f"{cart_res['test_r2']:.4f}")

        # Tree visualization
        st.markdown("#### Best CART Tree (top 3 levels)")
        try:
            best_cart = joblib.load(f"{ARTIFACT_DIR}/model_cart_tuned.joblib")
            fig_tree, ax = plt.subplots(figsize=(20, 10))
            plot_tree(best_cart, feature_names=feature_cols, filled=True,
                      rounded=True, max_depth=3, fontsize=8, ax=ax)
            plt.tight_layout()
            st.pyplot(fig_tree)
            plt.close()
        except Exception as e:
            st.info(f"Tree visualization not available: {e}")

        # Predicted vs actual for CART
        if 'CART (Tuned)' in predictions_p2:
            y_pred_cart = predictions_p2['CART (Tuned)']
            sample_idx = np.random.RandomState(42).choice(len(y_test_part2), min(5000, len(y_test_part2)), replace=False)
            fig = px.scatter(x=y_test_part2[sample_idx], y=y_pred_cart[sample_idx], opacity=0.3,
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title='CART: Predicted vs Actual (Test Set)')
            max_val = max(y_test_part2.max(), y_pred_cart.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                     line=dict(dash='dash', color='red'), name='Perfect'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ── 2.4 RANDOM FOREST ──
        st.markdown("---")
        st.markdown("### 2.4 Random Forest — GridSearchCV")
        rf_bp = grid_params.get('rf', {})
        st.markdown(f"**Best hyperparameters:** `n_estimators={rf_bp.get('n_estimators')}`, "
                    f"`max_depth={rf_bp.get('max_depth')}`")
        st.markdown("**Param grid:** n_estimators=[50,100,200], max_depth=[3,5,8] → 9 combos × 5 folds")

        try:
            rf_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_rf_cv_results.json")
            pivot_rf = rf_cv.pivot_table(values='mean_test_score',
                          index='param_max_depth', columns='param_n_estimators')
            fig = px.imshow(np.sqrt(-pivot_rf), text_auto='.3f', color_continuous_scale='RdYlGn_r',
                            title='RF GridSearchCV: Mean CV RMSE',
                            labels=dict(x='n_estimators', y='max_depth', color='RMSE'))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        rf_res = test_results_p2.get('Random Forest (Tuned)', {})
        if rf_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{rf_res['test_rmse']:.4f}")
            c2.metric("Test MAE", f"{rf_res['test_mae']:.4f}")
            c3.metric("Test R²", f"{rf_res['test_r2']:.4f}")

        if 'Random Forest (Tuned)' in predictions_p2:
            y_pred_rf = predictions_p2['Random Forest (Tuned)']
            sample_idx = np.random.RandomState(42).choice(len(y_test_part2), min(5000, len(y_test_part2)), replace=False)
            fig = px.scatter(x=y_test_part2[sample_idx], y=y_pred_rf[sample_idx], opacity=0.3,
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title='Random Forest: Predicted vs Actual (Test Set)')
            max_val = max(y_test_part2.max(), y_pred_rf.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                     line=dict(dash='dash', color='red'), name='Perfect'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ── 2.5 LIGHTGBM ──
        st.markdown("---")
        st.markdown("### 2.5 LightGBM — GridSearchCV (3 hyperparameters)")
        lgb_bp = grid_params.get('lgb', {})
        st.markdown(f"**Best hyperparameters:** `n_estimators={lgb_bp.get('n_estimators')}`, "
                    f"`max_depth={lgb_bp.get('max_depth')}`, `learning_rate={lgb_bp.get('learning_rate')}`")
        st.markdown("**Param grid:** n_estimators=[50,100,200], max_depth=[3,4,5,6], learning_rate=[0.01,0.05,0.1] → 36 combos × 5 folds")

        try:
            lgb_cv = pd.read_json(f"{ARTIFACT_DIR}/grid_lgb_cv_results.json")
            for lr in [0.01, 0.05, 0.1]:
                subset = lgb_cv[lgb_cv['param_learning_rate'] == lr]
                if len(subset) > 0:
                    pivot_lgb = subset.pivot_table(values='mean_test_score',
                                   index='param_max_depth', columns='param_n_estimators')
                    fig = px.imshow(np.sqrt(-pivot_lgb), text_auto='.3f', color_continuous_scale='RdYlGn_r',
                                    title=f'LightGBM: RMSE (lr={lr})',
                                    labels=dict(x='n_estimators', y='max_depth', color='RMSE'))
                    fig.update_layout(height=280)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        lgb_res = test_results_p2.get('LightGBM (Tuned)', {})
        if lgb_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{lgb_res['test_rmse']:.4f}")
            c2.metric("Test MAE", f"{lgb_res['test_mae']:.4f}")
            c3.metric("Test R²", f"{lgb_res['test_r2']:.4f}")

        if 'LightGBM (Tuned)' in predictions_p2:
            y_pred_lgb = predictions_p2['LightGBM (Tuned)']
            sample_idx = np.random.RandomState(42).choice(len(y_test_part2), min(5000, len(y_test_part2)), replace=False)
            fig = px.scatter(x=y_test_part2[sample_idx], y=y_pred_lgb[sample_idx], opacity=0.3,
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title='LightGBM: Predicted vs Actual (Test Set)')
            max_val = max(y_test_part2.max(), y_pred_lgb.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                     line=dict(dash='dash', color='red'), name='Perfect'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ── 2.6 MLP ──
        st.markdown("---")
        st.markdown("### 2.6 MLP Neural Network")
        st.markdown("""
        | Layer | Units | Activation |
        |-------|-------|------------|
        | Input | 21 (standardized) | — |
        | Hidden 1 | 128 | ReLU |
        | Hidden 2 | 128 | ReLU |
        | Output | 1 | Linear |

        **Loss:** MSE | **Optimizer:** Adam (lr=0.001) | **Early Stopping:** patience=10 | **Batch Size:** 256
        """)

        if mlp_history:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=mlp_history['loss'], name='Train Loss', mode='lines'))
                fig.add_trace(go.Scatter(y=mlp_history['val_loss'], name='Val Loss', mode='lines'))
                fig.update_layout(title='Loss Curve (MSE)', xaxis_title='Epoch', yaxis_title='MSE', height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=mlp_history['mae'], name='Train MAE', mode='lines'))
                fig.add_trace(go.Scatter(y=mlp_history['val_mae'], name='Val MAE', mode='lines'))
                fig.update_layout(title='MAE Curve', xaxis_title='Epoch', yaxis_title='MAE', height=350)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Epochs trained:** {len(mlp_history['loss'])} (early stopping)")

        mlp_res = test_results_p2.get('MLP', {})
        if mlp_res:
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{mlp_res['test_rmse']:.4f}")
            c2.metric("Test MAE", f"{mlp_res['test_mae']:.4f}")
            c3.metric("Test R²", f"{mlp_res['test_r2']:.4f}")

        if 'MLP' in predictions_p2:
            y_pred_mlp = predictions_p2['MLP']
            sample_idx = np.random.RandomState(42).choice(len(y_test_part2), min(5000, len(y_test_part2)), replace=False)
            fig = px.scatter(x=y_test_part2[sample_idx], y=y_pred_mlp[sample_idx], opacity=0.3,
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title='MLP: Predicted vs Actual (Test Set)')
            max_val = max(y_test_part2.max(), y_pred_mlp.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                     line=dict(dash='dash', color='red'), name='Perfect'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ── 2.7 COMPARISON SUMMARY ──
        st.markdown("---")
        st.markdown("### 2.7 Model Comparison Summary")

        summary_rows = []
        for name, res in test_results_p2.items():
            summary_rows.append({'Model': name, 'Test RMSE': res['test_rmse'],
                                 'Test MAE': res['test_mae'], 'Test R²': res['test_r2']})
        summary_df = pd.DataFrame(summary_rows).sort_values('Test RMSE')

        st.dataframe(
            summary_df.style
                .highlight_min(subset=['Test RMSE', 'Test MAE'], color='#c6efce')
                .highlight_max(subset=['Test R²'], color='#c6efce')
                .format({'Test RMSE': '{:.4f}', 'Test MAE': '{:.4f}', 'Test R²': '{:.4f}'}),
            use_container_width=True
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(summary_df, x='Model', y='Test RMSE', color='Test RMSE',
                         color_continuous_scale='RdYlGn_r',
                         title='Test-Set RMSE (Lower is Better)')
            fig.update_layout(height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(summary_df, x='Model', y='Test R²', color='Test R²',
                         color_continuous_scale='RdYlGn',
                         title='Test-Set R² (Higher is Better)')
            fig.update_layout(height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        best = summary_df.iloc[0]
        st.markdown(f"""
        **{best['Model']}** achieves the lowest test-set RMSE ({best['Test RMSE']:.4f}) and highest
        R² ({best['Test R²']:.4f}), confirming gradient-boosted trees' ability to capture non-linear
        interactions. The **MLP** provides competitive performance, demonstrating deep learning's
        viability on tabular data. **Linear models** serve as interpretable baselines but are outperformed
        by all non-linear models. **Trade-offs:** Linear models are fastest and most interpretable;
        CART offers visual interpretability via tree diagrams; RF and LightGBM offer the best accuracy
        but require SHAP for explainability; the MLP is a flexible black-box requiring longer training.
        """)


# ══════════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION
# ══════════════════════════════════════════════════════════
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    # ── SHAP INTERPRETATION ──
    st.markdown("""
    ### SHAP Interpretation

    **Strongest predictors:** The binary skill-presence flags — `has_management`
    (mean |SHAP| ≈ 1.7–1.9), `has_data` (≈ 1.6–1.8), `has_communication` (≈ 1.1–1.3), and
    `has_leadership` (≈ 1.0–1.1) — are the most impactful features. Postings that mention
    broad, cross-functional skills tend to list more total skills.

    **Direction of impact:** All `has_*` flags have a **positive** SHAP effect when present
    (flag = 1) and near-zero when absent. `avg_skill_length` pushes predictions upward for
    longer skill names. Geographic and temporal features have weak effects.

    **Decision-maker insights:** Recruiters should expect longer skill lists for postings
    mentioning management/data/communication. Job platforms could auto-suggest additional skills
    based on these signals. The dominance of soft-skill flags over technical ones indicates
    that role complexity, not specialization, drives skill breadth.
    """)

    # ── SHAP PLOTS ──
    st.markdown("---")
    selected_shap_model = st.selectbox("Select model for SHAP analysis:", list(DISPLAY_TO_SAFE.keys()))
    safe_name = DISPLAY_TO_SAFE[selected_shap_model]
    sv = shap_values_dict[safe_name]
    is_linear = selected_shap_model in ['Linear Regression', 'Lasso', 'Ridge']
    X_display = X_shap_scaled if is_linear else X_shap

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Summary Plot (Beeswarm)")
        fig_shap, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_display, feature_names=feature_cols, show=False, max_display=21)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

    with col2:
        st.markdown("### Bar Plot (Mean |SHAP|)")
        fig_bar, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_display, feature_names=feature_cols, plot_type='bar',
                          show=False, max_display=21)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()

    st.markdown("---")
    st.markdown("### Waterfall Plot (Individual Prediction)")
    sample_idx_shap = st.slider("Select sample index:", 0, len(sv)-1, 0)
    base_value = models[selected_shap_model].predict(X_display).mean() if is_linear else models[selected_shap_model].predict(X_shap).mean()
    explanation = shap.Explanation(
        values=sv[sample_idx_shap], base_values=base_value,
        data=X_display[sample_idx_shap], feature_names=feature_cols
    )
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(f'Waterfall — Sample #{sample_idx_shap}', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # ── INTERACTIVE PREDICTION ──
    st.markdown("---")
    st.markdown("### 🎯 Interactive Prediction")
    st.markdown("""
    <div class="callout-box" style="border-left-color: #FFA15A;">
        Set feature values below and see what the model predicts in real time. Adjust sliders,
        dropdowns, and checkboxes, then scroll down to see the predicted skill count and SHAP explanation.
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    pred_model_name = st.selectbox("Choose prediction model:", list(models.keys()), index=5)  # default LightGBM
    pred_model = models[pred_model_name]
    pred_is_linear = pred_model_name in ['Linear Regression', 'Lasso', 'Ridge']

    # Feature inputs — select the most meaningful features for manual setting
    st.markdown("#### Set Feature Values")
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        input_job_level = st.selectbox("Job Level", ['Associate', 'Mid-Senior'], index=1)
        input_job_type = st.selectbox("Job Type", ['Onsite', 'Hybrid', 'Remote'], index=0)
        input_country = st.selectbox("Country", ['United States', 'Canada', 'United Kingdom', 'Australia'], index=0)
        input_title_length = st.slider("Job Title Length (chars)", 5, 150, 40)

    with col2:
        input_title_words = st.slider("Title Word Count", 1, 20, 5)
        input_company_len = st.slider("Company Name Length", 2, 80, 15)
        input_avg_skill_len = st.slider("Avg Skill Name Length", 3.0, 30.0, 12.0, 0.5)
        input_is_top = st.selectbox("Top Company?", [0, 1], index=0)

    with col3:
        input_has_comm = st.checkbox("Has Communication", value=True)
        input_has_lead = st.checkbox("Has Leadership", value=False)
        input_has_mgmt = st.checkbox("Has Management", value=True)
        input_has_python = st.checkbox("Has Python", value=False)
        input_has_excel = st.checkbox("Has Excel", value=False)
        input_has_sales = st.checkbox("Has Sales", value=False)
        input_has_mktg = st.checkbox("Has Marketing", value=False)
        input_has_data = st.checkbox("Has Data", value=False)
        input_has_pm = st.checkbox("Has Project Management", value=False)
        input_has_cust = st.checkbox("Has Customer", value=False)

    st.markdown('</div>', unsafe_allow_html=True)  # close input-panel

    # Encode categorical inputs using label encoders
    le_level = label_encoders.get('job_level', None)
    le_type = label_encoders.get('job_type', None)
    le_country = label_encoders.get('search_country', None)

    try:
        level_enc = le_level.transform([input_job_level])[0] if le_level else 0
    except ValueError:
        level_enc = 0
    try:
        type_enc = le_type.transform([input_job_type])[0] if le_type else 0
    except ValueError:
        type_enc = 0
    try:
        country_enc = le_country.transform([input_country])[0] if le_country else 0
    except ValueError:
        country_enc = 0

    # Use median state encoding as default
    state_enc = int(np.median(df['state_encoded']))
    day_of_week = 2  # Wednesday
    month_val = int(df['month'].mode().iloc[0]) if len(df['month'].mode()) > 0 else 1

    # Build feature vector in the exact order of feature_cols
    input_vector = np.array([[
        level_enc,              # job_level_encoded
        type_enc,               # job_type_encoded
        country_enc,            # search_country_encoded
        state_enc,              # state_encoded
        input_title_length,     # title_length
        input_title_words,      # title_word_count
        input_company_len,      # company_name_length
        day_of_week,            # day_of_week
        month_val,              # month
        input_avg_skill_len,    # avg_skill_length
        int(input_is_top),      # is_top_company
        int(input_has_comm),    # has_communication
        int(input_has_lead),    # has_leadership
        int(input_has_mgmt),    # has_management
        int(input_has_python),  # has_python
        int(input_has_excel),   # has_excel
        int(input_has_sales),   # has_sales
        int(input_has_mktg),    # has_marketing
        int(input_has_data),    # has_data
        int(input_has_pm),      # has_project_management
        int(input_has_cust),    # has_customer
    ]], dtype=float)

    # Scale if linear model
    if pred_is_linear:
        input_for_pred = scaler.transform(input_vector)
    else:
        input_for_pred = input_vector

    # Predict
    prediction = pred_model.predict(input_for_pred)[0]

    st.markdown("---")
    st.markdown("### Prediction Result")
    above_avg = prediction > y.mean()
    diff = abs(prediction - y.mean())
    grad_color = "linear-gradient(135deg, #00CC96 0%, #059669 100%)" if above_avg else "linear-gradient(135deg, #636EFA 0%, #4854d4 100%)"
    direction_text = "above" if above_avg else "below"
    st.markdown(f"""
    <div class="prediction-card" style="background: {grad_color};">
        <div class="pred-label">Predicted Number of Skills ({pred_model_name})</div>
        <div class="pred-value">{prediction:.1f}</div>
        <div class="pred-subtitle">
            Dataset average: {y.mean():.1f} skills &nbsp;&bull;&nbsp;
            This prediction is <strong>{direction_text}</strong> average by {diff:.1f} skills
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP waterfall for this custom input
    st.markdown("---")
    st.markdown("### SHAP Waterfall for Your Custom Input")

    shap_safe = DISPLAY_TO_SAFE.get(pred_model_name, 'lightgbm')
    try:
        if pred_is_linear:
            explainer = shap.LinearExplainer(pred_model, X_scaled)
            shap_vals = explainer.shap_values(input_for_pred)
        else:
            explainer = shap.TreeExplainer(pred_model)
            shap_vals = explainer.shap_values(input_for_pred)

        custom_explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
            data=input_for_pred[0] if pred_is_linear else input_vector[0],
            feature_names=feature_cols
        )
        shap.waterfall_plot(custom_explanation, show=False, max_display=15)
        plt.title(f'SHAP Waterfall — Custom Input ({pred_model_name})', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    except Exception as e:
        st.info(f"SHAP waterfall for custom input not available: {e}")
